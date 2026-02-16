import asyncio
import base64
import io
import json
import os
import time
import re
import html
import traceback
from urllib.parse import urlsplit, urlunsplit, quote
from collections import defaultdict
from fastapi import FastAPI
import uvicorn
import asyncio

import qrcode
from qrcode.exceptions import DataOverflowError

from aiogram import Bot, Dispatcher, F, BaseMiddleware
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import (
    Message,
    CallbackQuery,
    BotCommand,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
)

# ================== تنظیمات ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
PROXY = os.getenv("PROXY", "")  # خالی یعنی بدون پروکسی

OWNER_ID = int(os.getenv("OWNER_ID"))
ADMINS = set(map(int, os.getenv("ADMINS", "").split(",")))

ADMIN_ORDER = [OWNER_ID] + sorted(list(ADMINS))
ADMIN_LABELS = {uid: f"کانفیگای ادمین {i+1}" for i, uid in enumerate(ADMIN_ORDER)}

ALLOWED_USERS = {OWNER_ID, *ADMINS}
MAX_TEXT_LEN = 3999
STORE_PATH = "configs_store.json"
# ============================================

# ================== Anti-Duplicate Guard ==================
_LAST_HANDLED_NAME_MSG = {}  # chat_id -> message_id
_LAST_HANDLED_CB = {}        # user_id -> last_cb_id


def guard_cb(cb: CallbackQuery) -> bool:
    """
    اگر همین Callback دوبار رسید، اجرا نشه.
    True یعنی قبلا هندل شده و باید return کنی.
    """
    uid = cb.from_user.id
    last = _LAST_HANDLED_CB.get(uid)
    if last == cb.id:
        return True
    _LAST_HANDLED_CB[uid] = cb.id
    return False


# ================== Middleware ادمین ==================
class OnlyAllowedMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        # هم پیام و هم کال‌بک رو چک کن
        if isinstance(event, Message):
            if event.from_user and event.from_user.id not in ALLOWED_USERS:
                await event.answer(
                    "⛔ شما اجازه استفاده از این ربات را ندارید.\n"
                    "برای پشتیبانی و خرید اشتراک با ایدی @Game_centerZ در ارتباط باشید."
                )
                return
        if isinstance(event, CallbackQuery):
            if event.from_user and event.from_user.id not in ALLOWED_USERS:
                await event.answer("⛔ دسترسی ندارید", show_alert=True)
                return
        return await handler(event, data)


# ================== FSM ==================
class Form(StatesGroup):
    waiting_links = State()
    waiting_name = State()


# ================== داشبورد (ReplyKeyboard) ==================
def dashboard_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📥 ارسال کانفیگ ها")],
            [KeyboardButton(text="👥 کاربرها")],
            [KeyboardButton(text="🔙 بازگشت / لغو")],
            [KeyboardButton(text="🏠 منوی اصلی")],
        ],
        resize_keyboard=True,
    )


def confirm_name_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ ادامه (تغییر نام)", callback_data="CONFIRM_NAME")],
            [InlineKeyboardButton(text="❌ لغو", callback_data="CANCEL_NAME")],
        ]
    )


def group_options_kb(user_id: int, base: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🗑 حذف همین مجموعه", callback_data=f"DG:{user_id}:{base}")],
            [
                InlineKeyboardButton(text="⬅️ برگشت", callback_data=f"U:{user_id}"),
                InlineKeyboardButton(text="🏠 صفحه اصلی", callback_data="HOME"),
            ],
        ]
    )


def confirm_delete_group_kb(user_id: int, base: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ تایید حذف", callback_data=f"DG_OK:{user_id}:{base}"),
                InlineKeyboardButton(text="❌ لغو حذف", callback_data=f"DG_NO:{user_id}:{base}"),
            ],
            [
                InlineKeyboardButton(text="⬅️ برگشت", callback_data=f"U:{user_id}"),
                InlineKeyboardButton(text="🏠 صفحه اصلی", callback_data="HOME"),
            ],
        ]
    )


# ================== ابزارهای کمکی ==================
def extract_links_from_text(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("vmess://") or line.startswith("vless://"):
            out.append(line)

    if not out:
        t = (text or "").strip()
        if t.startswith(("vmess://", "vless://")):
            out.append(t)

    return out


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    res = []
    for x in items:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


async def send_copyable_pre(message: Message, text: str) -> None:
    """
    متن رو داخل <pre> می‌فرسته تا تلگرام دکمه Copy بده.
    اگر طول زیاد بود، تکه‌تکه می‌کنه.
    """
    raw = text or ""
    max_chunk = 3500  # امن برای HTML + محدودیت‌های تلگرام
    i = 0
    while i < len(raw):
        chunk = raw[i:i + max_chunk]
        i += max_chunk
        await message.answer(f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML")


def make_qr_png_bytes(text: str) -> bytes:
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image()
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


async def send_qr_photo(
    message: Message,
    qr_text: str,
    title: str,
    links_text: str | None = None,
    per_group: int = 3,   # 2 یا 3
):
    """
    ✅ لینک‌ها داخل کپشن هم یکجا قابل کپی میشن (با <pre>)
    ✅ اگر جا نشد، ادامه لینک‌ها تو پیام بعدی 2تا/3تا داخل یک pre میاد (همه با هم کپی میشن)
    """
    png = make_qr_png_bytes(qr_text)
    file = BufferedInputFile(png, filename="qr.png")

    if not links_text:
        await message.answer_photo(file, caption=title[:1024])
        return

    links = [ln.strip() for ln in links_text.splitlines() if ln.strip()]
    remaining = []

    header = f"{title}\n\n🔗 لینک‌ها (قابل کپی):\n"
    pre_open = "<pre>"
    pre_close = "</pre>"

    cap_limit = 1024
    available = cap_limit - len(header) - len(pre_open) - len(pre_close)
    if available < 0:
        available = 0

    chosen = []
    used = 0
    for ln in links:
        piece = ln + "\n"
        piece_len = len(html.escape(piece))
        if used + piece_len <= available:
            chosen.append(ln)
            used += piece_len
        else:
            remaining.append(ln)

    caption = header + pre_open + html.escape("\n".join(chosen)) + pre_close
    await message.answer_photo(file, caption=caption[:1024], parse_mode="HTML")

    # ادامه در پیام‌های بعدی 2تا/3تا
    if remaining:
        await message.answer("ادامه لینک‌ها 👇")
        for i in range(0, len(remaining), per_group):
            block = "\n".join(remaining[i:i + per_group])
            await message.answer(f"<pre>{html.escape(block)}</pre>", parse_mode="HTML")


# ================== Storage (JSON file) ==================
def _load_store() -> dict:
    if not os.path.exists(STORE_PATH):
        return {"users": {}}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}}


def _save_store(data: dict) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def store_add_configs(user_id: int, username: str, full_name: str, items: list[dict]) -> None:
    db = _load_store()
    users = db.setdefault("users", {})

    uid = str(user_id)
    u = users.setdefault(uid, {
        "user_id": user_id,
        "username": username or "",
        "full_name": full_name or "",
        "items": []
    })

    u["username"] = username or u.get("username", "")
    u["full_name"] = full_name or u.get("full_name", "")

    now = int(time.time())
    for it in items:
        u["items"].append({
            "name": it["name"],
            "link": it["link"],
            "ts": now
        })

    _save_store(db)


def store_delete_item(user_id: int, ts: int) -> bool:
    db = _load_store()
    users = db.get("users", {})
    uid = str(user_id)

    if uid not in users:
        return False

    items = users[uid].get("items", [])
    before = len(items)
    users[uid]["items"] = [it for it in items if int(it.get("ts", 0)) != int(ts)]
    _save_store(db)
    return len(users[uid]["items"]) != before


# ================== Admin Inline Keyboards ==================
def users_list_kb() -> InlineKeyboardMarkup:
    db = _load_store()
    users = list(db.get("users", {}).values())

    def last_ts(u):
        items = u.get("items", [])
        return items[-1]["ts"] if items else 0

    users.sort(key=last_ts, reverse=True)

    rows = []
    if not users:
        rows.append([InlineKeyboardButton(text="❌ هنوز چیزی ذخیره نشده", callback_data="NOOP")])
    else:
        for u in users[:80]:
            uid = u.get("user_id")
            title = ADMIN_LABELS.get(uid, f"کانفیگای ادمین {uid}")
            rows.append([InlineKeyboardButton(text=f"👤 {title}", callback_data=f"U:{uid}")])

    rows.append([InlineKeyboardButton(text="🏠 صفحه اصلی", callback_data="HOME")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def user_configs_kb(user_id: int) -> InlineKeyboardMarkup:
    db = _load_store()
    u = db.get("users", {}).get(str(user_id))
    items = (u or {}).get("items", [])

    items = list(items)[-300:]
    items.reverse()

    groups = defaultdict(list)

    for it in items:
        name = it.get("name", "")
        m = re.match(r"^(.*?)-(\d+)$", name)
        if m:
            base = m.group(1).strip()
            num = int(m.group(2))
        else:
            base = name.strip()
            num = None
        groups[base].append({"num": num, "it": it})

    rows = []
    if not groups:
        rows.append([InlineKeyboardButton(text="❌ کانفیگی ذخیره نشده", callback_data="NOOP")])
    else:
        def group_last_ts(base):
            lst = groups[base]
            return max(x["it"].get("ts", 0) for x in lst)

        bases_sorted = sorted(groups.keys(), key=group_last_ts, reverse=True)

        for base in bases_sorted[:80]:
            lst = groups[base]
            nums = sorted([x["num"] for x in lst if x["num"] is not None])
            if nums:
                nums_text = ",".join(map(str, nums))
                title = f"📌 {base} - ({nums_text})"
            else:
                title = f"📌 {base}"

            rows.append([InlineKeyboardButton(text=title, callback_data=f"G:{user_id}:{base[:30]}")])

    rows.append([
        InlineKeyboardButton(text="⬅️ برگشت", callback_data="BACK_USERS"),
        InlineKeyboardButton(text="🏠 صفحه اصلی", callback_data="HOME"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ================== VMESS ==================
def _b64_decode(s: str) -> bytes:
    s = s.strip()
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode())


def _b64_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def parse_vmess(uri: str) -> dict:
    payload = uri[len("vmess://"):].strip()
    return json.loads(_b64_decode(payload).decode("utf-8", errors="ignore"))


def build_vmess(data: dict) -> str:
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "vmess://" + _b64_encode(raw)


def rename_vmess(uri: str, new_name: str) -> str:
    data = parse_vmess(uri)
    data["ps"] = new_name
    return build_vmess(data)


# ================== VLESS ==================
def rename_vless(uri: str, new_name: str) -> str:
    parts = urlsplit(uri)
    safe_fragment = quote(new_name, safe="")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, safe_fragment))


# ================== Bot ==================
dp = Dispatcher()
dp.message.middleware(OnlyAllowedMiddleware())
dp.callback_query.middleware(OnlyAllowedMiddleware())


@dp.error()
async def global_error_handler(event):
    try:
        print("🔥 ERROR:", repr(event.exception))
        traceback.print_exception(type(event.exception), event.exception, event.exception.__traceback__)
    except Exception:
        pass


# ================== /start ==================
@dp.message(CommandStart())
async def start_handler(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "سلام به Game/Vpn Center خوش آمدید 🌹\n"
        "از داشبورد گزینه مورد نظر را انتخاب نمایید🙏🏻",
        reply_markup=dashboard_kb()
    )


# -------- داشبورد: کاربرها (فقط ادمین) --------
@dp.message(F.text == "👥 کاربرها")
async def admin_users_menu(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("👥 لیست کاربران:", reply_markup=users_list_kb())


@dp.callback_query(F.data == "HOME")
async def cb_home(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer()
    await state.clear()
    await cb.message.answer("🏠 صفحه اصلی\nگزینه مورد نظر را انتخاب کنید:", reply_markup=dashboard_kb())


@dp.callback_query(F.data == "BACK_USERS")
async def cb_back_users(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer()
    await cb.message.edit_text("👥 لیست کاربران:", reply_markup=users_list_kb())


@dp.callback_query(F.data.startswith("U:"))
async def cb_pick_user(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer()
    user_id = int(cb.data.split(":")[1])

    db = _load_store()
    u = db.get("users", {}).get(str(user_id))
    title = (u or {}).get("full_name") or (u or {}).get("username") or str(user_id)

    await cb.message.edit_text(
        f"📂 کانفیگ‌های {title}:\n(روی اسم بزن)",
        reply_markup=user_configs_kb(user_id)
    )


@dp.callback_query(F.data == "NOOP")
async def cb_noop(cb: CallbackQuery):
    await cb.answer()


# ================== حذف تک کانفیگ ==================
@dp.callback_query(F.data.startswith("DELITEM:"))
async def cb_delitem_ask(cb: CallbackQuery):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, ts_s = cb.data.split(":")
    user_id = int(uid_s)
    ts = int(ts_s)

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ تایید حذف", callback_data=f"DELITEM_OK:{user_id}:{ts}"),
            InlineKeyboardButton(text="❌ لغو حذف", callback_data=f"DELITEM_NO:{user_id}:{ts}"),
        ],
        [InlineKeyboardButton(text="⬅️ برگشت", callback_data=f"U:{user_id}")],
    ])

    await cb.message.answer("⚠️ مطمئنی می‌خوای همین کانفیگ حذف بشه؟", reply_markup=kb)


@dp.callback_query(F.data.startswith("DELITEM_OK:"))
async def cb_delitem_ok(cb: CallbackQuery):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, ts_s = cb.data.split(":")
    user_id = int(uid_s)
    ts = int(ts_s)

    ok = store_delete_item(user_id, ts)
    if ok:
        await cb.message.answer("✅ همین کانفیگ حذف شد.")
    else:
        await cb.message.answer("❌ پیدا نشد (شاید قبلاً حذف شده).")

    await cb.message.answer("📂 لیست کانفیگ‌ها:", reply_markup=user_configs_kb(user_id))


@dp.callback_query(F.data.startswith("DELITEM_NO:"))
async def cb_delitem_no(cb: CallbackQuery):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, _ = cb.data.split(":")
    user_id = int(uid_s)

    await cb.message.answer("✅ لغو شد.")
    await cb.message.answer("📂 لیست کانفیگ‌ها:", reply_markup=user_configs_kb(user_id))


# ================== نمایش QR تک کانفیگ (اگر جای دیگه داری استفاده کن) ==================
@dp.callback_query(F.data.startswith("C:"))
async def cb_send_config_qr(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, idx_s = cb.data.split(":")
    user_id = int(uid_s)
    idx = int(idx_s)

    db = _load_store()
    u = db.get("users", {}).get(str(user_id))
    items = (u or {}).get("items", [])
    items = list(items)[-120:]
    items.reverse()

    if idx < 0 or idx >= len(items):
        await cb.message.answer("❌ این مورد پیدا نشد یا حذف شده.")
        return

    it = items[idx]
    name = it["name"]
    link = it["link"]
    ts = int(it.get("ts", 0))

    caption = f"✅ {name}\n\n🔗 لینک:\n{link}"

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🗑 حذف همین کانفیگ", callback_data=f"DELITEM:{user_id}:{ts}")],
        [InlineKeyboardButton(text="⬅️ برگشت", callback_data=f"U:{user_id}")],
        [InlineKeyboardButton(text="🏠 صفحه اصلی", callback_data="HOME")],
    ])

    try:
        await send_qr_photo(cb.message, qr_text=link, title=caption, links_text=link, per_group=3)
        await cb.message.answer("⬇️ گزینه‌ها:", reply_markup=kb)
    except DataOverflowError:
        await cb.message.answer("❌ این لینک داخل QR جا نشد.")
        await cb.message.answer(caption, reply_markup=kb)


# ================== ارسال گروه (همه لینک‌ها یکجا + ادامه 3تا3تا) ==================
@dp.callback_query(F.data.startswith("G:"))
async def cb_send_group(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, base_short = cb.data.split(":", 2)
    user_id = int(uid_s)

    db = _load_store()
    u = db.get("users", {}).get(str(user_id))
    items = (u or {}).get("items", [])
    items = list(items)[-300:]
    items.reverse()

    matched = []
    for it in items:
        name = it.get("name", "")
        m = re.match(r"^(.*?)-(\d+)$", name)
        base = (m.group(1).strip() if m else name.strip())
        if base.startswith(base_short):
            matched.append(it)

    if not matched:
        await cb.message.answer("❌ چیزی پیدا نشد.")
        return

    def get_num(it):
        name = it.get("name", "")
        m = re.match(r"^(.*?)-(\d+)$", name)
        return int(m.group(2)) if m else 999999

    matched.sort(key=get_num)

    links = [it["link"] for it in matched if it.get("link")]
    if not links:
        await cb.message.answer("❌ لینک‌ها خالی بود.")
        return

    body = "\n".join(links)

    base_real = re.match(r"^(.*?)-(\d+)$", matched[0]["name"])
    base_title = (base_real.group(1).strip() if base_real else matched[0]["name"])

    title = (
    f"🔗 GAME/VPN CENTER 🎮⚡️\n"
    f"📌 USER: {base_title} ({len(links)} )\n\n"
    f"🛜 کانفیگ های اختصاصی  با حجم‌های متنوع\n"
    f"💬چنل تلگرام و اطلاع رسانی: @vpncentera\n"
    f"📥 ثبت سفارش سریع: @Game_centerZ\n\n"
    f"🔴رضایت مشتریان اولویت ماست | سرویس کاملاً تضمینی♾️"
)

    # ✅ QR + کپشن (اگر جا نشد ادامه لینک‌ها تو پیام‌های بعدی میره)
    try:
        await send_qr_photo(
            cb.message,
            qr_text=body,
            title=title,
            links_text=body,
            per_group=3
        )
    except DataOverflowError:
        await cb.message.answer("❌ این مجموعه داخل QR جا نشد.")

    # ✅ تایتل کامل برای لینک‌های قابل کپی + خود لینک‌ها داخل <pre>
    copy_title = (
    f"🔗 GAME/VPN CENTER 🎮⚡️\n"
    f"📌 USER: {base_name} ({len(renamed)} )\n\n"
    f"🛜 کانفیگ های اختصاصی  با حجم‌های متنوع\n"
    f"💬چنل تلگرام و اطلاع رسانی: @vpncentera\n"
    f"📥 ثبت سفارش سریع: @Game_centerZ\n\n"
    f"🔴رضایت مشتریان اولویت ماست | سرویس کاملاً تضمینی♾️"
)

    await send_copyable_pre(cb.message, body)

    # ✅ داشبورد گزینه‌ها
    await cb.message.answer(
        "گزینه‌ها:",
        reply_markup=group_options_kb(user_id, base_short)
    )




# ================== حذف مجموعه ==================
@dp.callback_query(F.data.startswith("DG:"))
async def cb_delete_group_ask(cb: CallbackQuery):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, base = cb.data.split(":", 2)
    user_id = int(uid_s)

    await cb.message.answer(
        "⚠️ مطمئنی این *مجموعه* حذف بشه؟",
        reply_markup=confirm_delete_group_kb(user_id, base),
        parse_mode="Markdown"
    )


@dp.callback_query(F.data.startswith("DG_NO:"))
async def cb_delete_group_cancel(cb: CallbackQuery):
    if guard_cb(cb):
        return
    await cb.answer("لغو شد ✅")

    _, uid_s, base = cb.data.split(":", 2)
    user_id = int(uid_s)

    await cb.message.answer("گزینه‌ها:", reply_markup=group_options_kb(user_id, base))


@dp.callback_query(F.data.startswith("DG_OK:"))
async def cb_delete_group_ok(cb: CallbackQuery):
    if guard_cb(cb):
        return
    await cb.answer()

    _, uid_s, base = cb.data.split(":", 2)
    user_id = int(uid_s)

    db = _load_store()
    u = db.get("users", {}).get(str(user_id))
    if not u:
        await cb.message.answer("❌ کاربر پیدا نشد.")
        return

    before = len(u.get("items", []))
    new_items = []
    for it in u.get("items", []):
        name = (it.get("name") or "").strip()
        if name.startswith(base + "-") or name == base:
            continue
        new_items.append(it)

    u["items"] = new_items
    db["users"][str(user_id)] = u
    _save_store(db)

    removed = before - len(new_items)
    await cb.message.answer(f"✅ {removed} مورد از این مجموعه حذف شد.")
    await cb.message.answer("📂 لیست کانفیگ‌ها:", reply_markup=user_configs_kb(user_id))


# ================== داشبورد: ارسال کانفیگ ها ==================
@dp.message(F.text == "📥 ارسال کانفیگ ها")
async def menu_send_configs(message: Message, state: FSMContext):
    await state.clear()
    await state.update_data(raw_links=[])
    await state.set_state(Form.waiting_links)

    await message.answer(
        "📥 کانفیگ‌ها را ارسال کنید\n"
        "✅ پشتیبانی: vmess:// و vless://\n"
        "بین هر کانفیگ یک لاین فاصله بگذارید.\n\n"
        "بعد از دریافت، از شما نام جدید را می‌پرسم.",
        reply_markup=dashboard_kb()
    )


@dp.message(F.text == "🔙 بازگشت / لغو")
async def menu_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("✅ لغو شد. از منو انتخاب کنید:", reply_markup=dashboard_kb())


@dp.message(F.text == "🏠 منوی اصلی")
async def menu_home(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("🏠 منوی اصلی\nگزینه مورد نظر را انتخاب کنید:", reply_markup=dashboard_kb())


@dp.callback_query(F.data == "CANCEL_NAME")
async def cb_cancel_name(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer("لغو شد ✅")
    await state.clear()
    await cb.message.answer("✅ لغو شد. از منو انتخاب کن:", reply_markup=dashboard_kb())


# -------- دریافت لینک‌ها --------
@dp.message(Form.waiting_links)
async def handle_links(message: Message, state: FSMContext):
    text = message.text or ""
    links = extract_links_from_text(text)

    if not links:
        await message.answer("❌ لینک معتبر پیدا نکردم. فقط vmess:// یا vless:// بفرست.")
        return

    data = await state.get_data()
    prev = data.get("raw_links", [])
    merged = dedupe_keep_order(prev + links)
    await state.update_data(raw_links=merged)

    vmess_count = sum(1 for x in merged if x.startswith("vmess://"))
    vless_count = sum(1 for x in merged if x.startswith("vless://"))
    total = len(merged)

    await message.answer(
        f"✅ کانفیگ‌ها ذخیره شد.\n"
        f"VMESS: {vmess_count} | VLESS: {vless_count} | مجموع: {total}\n\n"
        "اگر آماده‌ای برای تغییر نام، روی دکمه ادامه بزن ✅",
        reply_markup=confirm_name_kb()
    )


# -------- تایید رفتن به نام --------
@dp.callback_query(F.data == "CONFIRM_NAME")
async def cb_confirm_name(cb: CallbackQuery, state: FSMContext):
    if guard_cb(cb):
        return
    await cb.answer()

    data = await state.get_data()
    raw_links = data.get("raw_links", [])
    if not raw_links:
        await cb.message.answer("❌ اول لینک‌ها رو بفرست.")
        return

    await state.set_state(Form.waiting_name)
    await cb.message.answer("✍️ عالی! حالا نام جدید کانفیگ را ارسال کنید:")


# -------- دریافت نام + خروجی + QR --------
@dp.message(Form.waiting_name)
async def handle_name(message: Message, state: FSMContext):
    chat_id = message.chat.id
    last_id = _LAST_HANDLED_NAME_MSG.get(chat_id)
    if last_id == message.message_id:
        return
    _LAST_HANDLED_NAME_MSG[chat_id] = message.message_id

    if not message.text:
        await message.answer("❌ لطفاً نام را به صورت متن ارسال کنید.")
        return

    base_name = (message.text or "").strip()
    if not base_name:
        await message.answer("❌ نام خالی است. یک نام بفرست.")
        return

    data = await state.get_data()
    raw_links = data.get("raw_links", [])
    if not raw_links:
        await message.answer("❌ هنوز کانفیگی ارسال نشده.")
        await state.set_state(Form.waiting_links)
        return

    # از state نام خارج شو که دوباره تریگر نشه
    await state.set_state(Form.waiting_links)

    renamed = []
    bad = 0

    for idx, link in enumerate(raw_links, start=1):
        new_name = f"{base_name}-{idx}"
        try:
            if link.startswith("vmess://"):
                renamed.append(rename_vmess(link, new_name))
            elif link.startswith("vless://"):
                renamed.append(rename_vless(link, new_name))
            else:
                bad += 1
        except Exception:
            bad += 1

    if not renamed:
        await message.answer("❌ هیچ کانفیگی قابل تبدیل نبود.")
        return

    body = "\n".join(renamed)
    nums = ",".join(str(i) for i in range(1, len(renamed) + 1))
    names_text = "\n".join([f"کانفیگ {i} : {base_name}-{i}" for i in range(1, len(renamed) + 1)])
    bad_note = f"\n\n⚠️ {bad} کانفیگ مشکل‌دار رد شد." if bad else ""

    await message.answer(
        "✅ کانفیگ‌های تغییر نام یافته آماده شد.\n"
        "📌 نام‌ها:\n"
        + names_text
        + bad_note
    )

    await send_copyable_pre(message, body)

    qr_title = (
    f"🔗 GAME/VPN CENTER 🎮⚡️\n"
    f"📌 USER: {base_name} ({len(renamed)} )\n\n"
    f"🛜 کانفیگ های اختصاصی  با حجم‌های متنوع\n"
    f"💬چنل تلگرام و اطلاع رسانی: @vpncentera\n"
    f"📥 ثبت سفارش سریع: @Game_centerZ\n\n"
    f"🔴رضایت مشتریان اولویت ماست | سرویس کاملاً تضمینی♾️"
)


    try:
        await send_qr_photo(message, qr_text=body, title=qr_title, links_text=body, per_group=3)
    except DataOverflowError:
        mid = len(renamed) // 2 or 1
        part1 = "\n".join(renamed[:mid])
        part2 = "\n".join(renamed[mid:])

        await send_qr_photo(
            message,
            qr_text=part1,
            title=f"🧩 QR بخش 1\n{base_name} (1 تا {mid})",
            links_text=part1,
            per_group=3
        )

        if part2.strip():
            try:
                await send_qr_photo(
                    message,
                    qr_text=part2,
                    title=f"🧩 QR بخش 2\n{base_name} ({mid+1} تا {len(renamed)})",
                    links_text=part2,
                    per_group=3
                )
            except DataOverflowError:
                await message.answer("⚠️ تعداد لینک‌ها خیلی زیاد بود؛ QRهای جدا جدا می‌فرستم:")
                for i, link in enumerate(renamed, start=1):
                    try:
                        await send_qr_photo(message, qr_text=link, title=f"✅ QR {i}\nکانفیگ: {base_name}-{i}", links_text=link)
                    except DataOverflowError:
                        await message.answer(f"❌ QR کانفیگ {i} جا نشد.\n🔗 لینک:\n{link}")

    # ذخیره
    try:
        stored_items = [{"name": f"{base_name}-{i}", "link": link} for i, link in enumerate(renamed, start=1)]
        store_add_configs(
            user_id=message.from_user.id,
            username=message.from_user.username or "",
            full_name=message.from_user.full_name or "",
            items=stored_items
        )
    except Exception as e:
        await message.answer(f"⚠️ ذخیره‌سازی انجام نشد: {repr(e)}")

    await state.update_data(raw_links=[])
    await message.answer("✅ تمام شد. از داشبورد انتخاب کن:", reply_markup=dashboard_kb())


# ================== Bot Commands (/start menu) ==================
async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="🏠 بازگشت به منوی اصلی"),
    ]
    await bot.set_my_commands(commands)


app = FastAPI()

@app.get("/")
def home():
    return {"status": "OK", "bot": "running"}

async def run_bot():
    session = AiohttpSession(proxy=PROXY) if PROXY else AiohttpSession()
    bot = Bot(token=BOT_TOKEN, session=session)
    await set_commands(bot)
    await dp.start_polling(bot)

@app.on_event("startup")
async def startup():
    asyncio.create_task(run_bot())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

