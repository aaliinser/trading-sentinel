#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — بوت إشارات حي (v5.7.1 Deriv Real-Time Edition)
الاستراتيجية: RSI(14) Wilder + BB(20,2.0) ddof=0 | 5m / 15m
المصدر الأساسي: Deriv API (LIVE TICKS)
المصدر الاحتياطي: Yahoo Finance (Fallback)
الإصلاح النهائي: توحيد قائمة الرموز لضمان عمل Deriv بدون أخطاء "Symbol Not Found".
"""
import os, sys, time, json, logging, asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
import requests
import websockets # مكتبة Deriv للاتصال اللحظي

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ──────────────────────────────────────
# ⚠️ مهم جداً: هذه هي رموز Deriv الدقيقة. لا تستخدم رموز ياهو هنا.
# إذا أردت تغيير الأزواج، عدّل هذه القائمة فقط.
DEFAULT_DERIV_SYMBOLS = [
    "frxUSDJPY",   # USD/JPY
    "frxEURAUD",   # EUR/AUD
    "frxUSDCHF",   # USD/CHF
    "frxEURCAD",   # EUR/CAD
    "frxCADJPY"    # CAD/JPY
]

# قراءة الرموز من متغير البيئة أو استخدام الافتراضي
SYMBOLS_RAW = os.getenv("DERIV_SYMBOLS_LIST", ",".join(DEFAULT_DERIV_SYMBOLS))
SYMBOLS = [s.strip() for s in SYMBOLS_RAW.split(",") if s.strip()]

# إعدادات المؤشرات الفنية (كما هي في الاستراتيجية الأصلية)
RSI_P = 14
BB_P = 20
BB_K = 2.0
RSI_HI = 75.0
RSI_LO = 25.0
EXPIRY_MIN = 15
STOP_AFTER_LOSSES = 3
STOP_HOURS = 4
STATE_FILE = "state_h2.json"

USER_TZ_OFFSET_H = 3
WIN_START_H = 9
BLACKOUT_END_H = 3
MONTH_START = "2026-09-15"
BREAKEVEN_WR = 52.63

# Anti-freeze timeouts
PENDING_SOFT_TIMEOUT_MIN = 35
PENDING_HARD_TIMEOUT_MIN = 60
STATE_CLEANUP_HOURS = 2

AR_DAYS = ["الاثنين","الثلاثاء","الأربعاء","الخميس",
           "الجمعة","السبت","الأحد"]
AR_MONTHS = ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
             "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

# --- DERIV CONFIGURATION ---
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "").strip() 

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("H2v57-Deriv-Fixed")

# ─── Safe accessors ──────────────────────────────────────
def safe_list(st, key):
    val = st.get(key)
    if not isinstance(val, list):
        st[key] = []
        return st[key]
    return val

def safe_dict(st, key):
    val = st.get(key)
    if not isinstance(val, dict):
        st[key] = {}
        return st[key]
    return val

def ar_day(ds):
    try:
        d = datetime.strptime(ds, "%Y-%m-%d")
        return AR_DAYS[d.weekday()]
    except Exception:
        return ""

def ar_month(m):
    return AR_MONTHS[m - 1]

def load_state():
    p = Path(STATE_FILE)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                st = json.load(f)
            pend = st.get("pending", [])
            if isinstance(pend, list) and pend:
                now = datetime.now(timezone.utc)
                fresh = []
                dropped = 0
                for p_item in pend:
                    try:
                        exp = datetime.fromisoformat(p_item["exp"])
                        age_min = (now - exp).total_seconds() / 60.0
                        if age_min <= STATE_CLEANUP_HOURS * 60:
                            fresh.append(p_item)
                        else:
                            dropped += 1
                    except Exception:
                        dropped += 1
                st["pending"] = fresh
                if dropped:
                    log.info(f"state cleanup: dropped {dropped} stale pending signal(s)")
            return st
        except Exception as e:
            log.error(f"load_state error: {e}")
            return {}
    return {}

def save_state(st):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, default=str)
    except Exception as e:
        log.error(f"save: {e}")

def day_obj(st):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = st.get("day", {})
    if d.get("date") != today:
        old_cl = d.get("cl", 0)
        d = {"date": today, "alerts": 0, "trades": 0,
             "wins": 0, "losses": 0, "cl": old_cl}
        st["day"] = d
    return d

def tg_send(text, reply_to=None):
    if not TG_TOKEN or not TG_CHAT:
        log.info("TG disabled")
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text,
               "parse_mode": "Markdown",
               "disable_web_page_preview": True}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    for a in range(3):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return r.json().get("result", {}).get("message_id")
            time.sleep(2*a+1)
        except Exception as e:
            log.warning(f"tg attempt {a}: {e}")
    return None

# ─── NEW: Robust Deriv Data Fetcher ──────────────────────
async def _fetch_deriv_async(symbol, count=220):
    """
    يجلب الشموع من Deriv API.
    الرمز يجب أن يكون بصيغة Deriv (مثل frxUSDJPY).
    """
    uri = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"
    
    try:
        async with websockets.connect(uri, ping_interval=20, close_timeout=5) as ws:
            # 1. Authorization (Optional but good practice)
            if DERIV_API_TOKEN:
                auth_msg = {"authorize": DERIV_API_TOKEN}
                await ws.send(json.dumps(auth_msg))
                resp_auth = json.loads(await ws.recv())
                if 'error' in resp_auth:
                    log.warning(f"Deriv Auth Warning: {resp_auth['error'].get('message')}")
            
            # 2. Request History
            req_msg = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": 300, # 5 minutes
                "count": count,
                "end": "latest"
            }
            await ws.send(json.dumps(req_msg))
            
            # Receive data
            raw_data = json.loads(await ws.recv())
            
            if 'error' in raw_data:
                err_msg = raw_data['error'].get('message', 'Unknown Error')
                raise ValueError(f"Deriv API Error [{symbol}]: {err_msg}")
                
            candles = raw_data.get('candles', [])
            if not candles:
                return None
                
            # Convert to Pandas DataFrame
            df = pd.DataFrame(candles)
            df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
            df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
            df.set_index('epoch', inplace=True)
            df.sort_index(inplace=True)
            
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df[['Open','High','Low','Close']].dropna()

    except Exception as e:
        log.error(f"Websocet Connection/Error for {symbol}: {e}")
        return None

def fetch_data_deriv(symbol):
    """Sync wrapper for async Deriv call"""
    if not DERIV_API_TOKEN:
        log.warning("DERIV_API_TOKEN is empty. Skipping Deriv.")
        return None
        
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_fetch_deriv_async(symbol))
        loop.close()
        return result
    except Exception as e:
        log.error(f"Async Wrapper Error for {symbol}: {e}")
        return None

def flatten_columns(df):
    OHLC = ("Open", "High", "Low", "Close")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c) for c in df.columns]
        return df
    df = df.copy()
    price_level = None
    for lvl in range(df.columns.nlevels):
        values = {str(v) for v in df.columns.get_level_values(lvl)}
        if values & set(OHLC):
            price_level = lvl
            break
    if price_level is None:
        price_level = df.columns.nlevels - 1
    df.columns = [str(c) for c in df.columns.get_level_values(price_level)]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df

def fetch5m(sym):
    """
    يحاول أولاً Deriv (Live). إذا فشل، يعود لـ yfinance (Delayed).
    sym هنا يجب أن يكون رمز Deriv (مثل frxUSDJPY).
    لتحويله لياهو نحتاج لإزالة frx وإضافة =X
    """
    # 1. Try Deriv First
    df_deriv = fetch_data_deriv(sym)
    if df_deriv is not None and len(df_deriv) >= 60:
        log.info(f"✅ Loaded {len(df_deriv)} candles from Deriv for {sym}")
        return df_deriv
        
    # 2. Fallback to yfinance
    log.warning(f"⚠️ Deriv failed for {sym}. Trying yfinance fallback...")
    try:
        import yfinance as yf
        # Transform Deriv Symbol to Yahoo Symbol: frxUSDJPY -> USDJPY=X
        yf_sym = sym.replace("frx", "") + "=X"
        
        for a in range(3):
            try:
                df = yf.Ticker(yf_sym).history(period="3d", interval="5m",
                                            auto_adjust=False, actions=False, timeout=20)
                if df is None or df.empty:
                    raise ValueError("empty")
                df = flatten_columns(df)
                required = ["Open", "High", "Low", "Close"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"missing columns: {missing}")
                df = df[required]
                df.index = pd.to_datetime(df.index, utc=True)
                df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
                now = pd.Timestamp.now(tz="UTC")
                df = df[df.index <= now]
                if not df.empty and df.index[-1] + pd.Timedelta(minutes=5) > now:
                    df = df.iloc[:-1]
                if not df.empty:
                     log.info(f"✅ Loaded {len(df)} candles from yfinance for {yf_sym}")
                     return df
            except Exception as e:
                log.warning(f"{yf_sym} yf fetch {a}: {e}")
                time.sleep(2*a+1)
    except ImportError:
        log.error("yfinance not installed for fallback.")
    except Exception as e:
        log.error(f"Fallback process error: {e}")
        
    return None

def indicators(df):
    c = df["Close"]
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / RSI_P
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=RSI_P).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=RSI_P).mean()
    denom = avg_gain + avg_loss
    rsi = 100.0 * avg_gain / denom
    rsi = rsi.where(denom > 0, 50.0)
    mid = c.rolling(BB_P, min_periods=BB_P).mean()
    sd = c.rolling(BB_P, min_periods=BB_P).std(ddof=0)
    bu = mid + BB_K * sd
    bl = mid - BB_K * sd
    return rsi, bu, bl, sd

def fmt_sym(s):
    # Display nicely: frxUSDJPY -> USD/JPY
    clean = s.replace("frx", "")
    if len(clean) == 6:
        return f"{clean[:3]}/{clean[3:]}"
    return s

def fmt_px(v):
    v = float(v)
    return f"{v:.3f}" if v > 50 else f"{v:.5f}"

def calc_urgency(cl, band, sdv, direction):
    if sdv <= 0:
        return "unknown", "unknown", "⚠️", "خطأ في حساب الانحراف"
    if direction == "CALL":
        if cl <= band * 1.0003:
            return "strong", "enter_now", "✅", "ادخل فورا خلال 60 ثانية\n(السعر عند الباند — فرصة مثالية)"
        elif cl <= band * 1.0015:
            return "mid", "enter_now", "⚡", "ادخل خلال 60 ثانية\n(السعر قريب من الباند)"
        else:
            return "far", "skip", "🚫", "تخطّ — السعر ابتعد عن الباند"
    else:
        if cl >= band * 0.9997:
            return "strong", "enter_now", "✅", "ادخل فورا خلال 60 ثانية\n(السعر عند الباند — فرصة مثالية)"
        elif cl >= band * 0.9985:
            return "mid", "enter_now", "⚡", "ادخل خلال 60 ثانية\n(السعر قريب من الباند)"
        else:
            return "far", "skip", "🚫", "تخطّ — السعر ابتعد عن الباند"

# ─── Resolve Pending Logic ───────────────────────────────
def resolve_pending(st):
    pend = safe_list(st, "pending")
    if not pend:
        return
    now = datetime.now(timezone.utc)
    done = []
    resolved_count = 0
    voided_count = 0
    forced_count = 0

    for p in pend:
        try:
            exp = datetime.fromisoformat(p["exp"])
        except Exception:
            done.append(p)
            forced_count += 1
            continue

        age_min = (now - exp).total_seconds() / 60.0

        if age_min > PENDING_HARD_TIMEOUT_MIN:
            log.warning(f"{p.get('sym')}: hard timeout ({age_min:.0f}m) — forced cleanup")
            done.append(p)
            forced_count += 1
            continue

        if now < exp + timedelta(seconds=60):
            continue

        # Fetch latest price to determine win/loss
        # Note: We use the same symbol logic. If it was stored as frx..., we keep it.
        df = fetch5m(p["sym"]) 
        exit_px = None

        if df is not None and not df.empty:
            # Find the candle closest to expiry time
            target_time = exp - timedelta(minutes=5) # Approximate alignment for 5m candles
            # Get last closed candle before or at target
            candidates = df[df.index <= target_time]
            if not candidates.empty:
                exit_px = float(candidates["Close"].iloc[-1])
            else:
                # Fallback to very last available if target is too far back/new
                 exit_px = float(df["Close"].iloc[-1])

        if exit_px is None:
            if age_min > PENDING_SOFT_TIMEOUT_MIN:
                log.warning(f"void pending {p['sym']} (no data after {age_min:.0f}m)")
                done.append(p)
                voided_count += 1
            continue

        entry = float(p["px"])
        if abs(exit_px - entry) < 0.00001: # Tie handling
            log.info(f"tie void {p['sym']}")
            done.append(p)
            voided_count += 1
            continue

        win = exit_px > entry if p["dr"] == "CALL" else exit_px < entry
        sent = datetime.fromisoformat(p["sent"])
        loc = sent + timedelta(hours=USER_TZ_OFFSET_H)

        sim_list = safe_list(st, "sim")
        sim_list.append({
            "dl": loc.strftime("%Y-%m-%d"),
            "hl": loc.hour,
            "sym": p["sym"], "dr": p["dr"],
            "v4": bool(p.get("v4", False)), 
            "win": bool(win),
            "px": entry, "ex": exit_px,
        })
        if len(sim_list) > 2000:
            st["sim"] = sim_list[-2000:]

        log.info(f"auto-resolved {p['sym']} {p['dr']} win={win}")
        resolved_count += 1

        rid = p.get("mid")
        try:
            rid = int(rid) if rid is not None else None
        except Exception:
            rid = None
        txt = (f"{'✅' if win else '❌'} *نتيجة تلقائية*: "
               f"{'ربحت' if win else 'خسرت'}\n"
               f"• الدخول: {fmt_px(entry)}\n"
               f"• الخروج: {fmt_px(exit_px)}\n"
               f"• ضمن V4: {'✅' if p.get('v4') else '➖'}")
        tg_send(txt, reply_to=rid)
        done.append(p)

    for p in done:
        if p in pend:
            pend.remove(p)

    if resolved_count or voided_count or forced_count:
        log.info(f"resolve_pending: resolved={resolved_count} voided={voided_count} forced={forced_count} remaining={len(pend)}")

# ─── Reporting Functions ─────────────────────────────────
def send_report(st):
    sim = safe_list(st, "sim")
    man = safe_list(st, "log")
    if not sim and not man:
        tg_send("📊 لا بيانات مسجلة بعد")
        return
    n = len(sim)
    w = sum(1 for x in sim if x["win"])
    wr = round(100*w/n, 1) if n else 0.0
    txt = (f"📊 *تقرير الأداء (v5.7.1 Deriv)*\n\n"
           f"*إشارات:* *{n}* | فوز *{wr}%*\n\n"
           f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def send_day_summary(st, day_str):
    sim = [x for x in safe_list(st, "sim") if x["dl"] == day_str]
    win_sig = [x for x in sim if x["hl"] >= WIN_START_H]
    n = len(win_sig)
    w = sum(1 for x in win_sig if x["win"])
    l = n - w
    wr = round(100*w/n, 1) if n else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    pnl = round(w * win_payout - l * stake, 2)
    txt = (f"🌙 *ملخص يوم {ar_day(day_str)} {day_str}*\n"
           f"• إشارات النافذة: *{n}*\n"
           f"• فوز {w} / خسارة {l} | *{wr}%*\n"
           f"• صافي تقديري: {pnl:+.2f}$\n\n"
           f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def listen(st):
    if not TG_TOKEN:
        return
    try:
        off = st.get("tg_offset", 0)
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                         params={"offset": off, "timeout": 0}, timeout=15)
        for u in r.json().get("result", []):
            uid = u.get("update_id", 0)
            if uid >= off:
                off = uid + 1
            m = u.get("message") or u.get("edited_message")
            if not m:
                continue
            t = (m.get("text") or "").lower()
            if ("تقرير" in t) or ("report" in t):
                send_report(st)
                continue
            win = None
            if any(w in t for w in ["ربحت","رابحة","won","win","ربح"]):
                win = True
            elif any(w in t for w in ["خسرت","خاسرة","lost","lose","خسارة"]):
                win = False
            if win is None:
                continue
            rt = m.get("reply_to_message") or {}
            mid = str(rt.get("message_id",""))
            op = safe_dict(st, "open").get(mid)
            d = day_obj(st)
            if op is None:
                tg_send("⚠️ رد على رسالة الإشارة مباشرة")
                continue
            del st["open"][mid]
            safe_list(st, "log").append({
                "date": d["date"],
                "sym": op.get("sym", "?"),
                "dr": op.get("dr", "?"),
                "v4": bool(op.get("v4", False)),
                "win": bool(win),
            })
            log_list = safe_list(st, "log")
            if len(log_list) > 600:
                st["log"] = log_list[-600:]
            d["trades"] += 1
            if win:
                d["wins"] += 1
                d["cl"] = 0
            else:
                d["losses"] += 1
                d["cl"] += 1
                if d["cl"] >= STOP_AFTER_LOSSES:
                    st["stop_until"] = time.time() + STOP_HOURS*3600
                    d["cl"] = 0
            wr = round(100*d["wins"]/d["trades"],1) if d["trades"] else 0
            win_payout = 6.0 * 0.90
            stake = 6.0
            pnl = round(d["wins"] * win_payout - d["losses"] * stake, 2)
            tg_send(f"💰 سُجلت: {'✅' if win else '❌'}\n"
                    f"• اليوم: {d['trades']} صفقة | فوز {wr}%\n"
                    f"• صافي تقديري: {pnl:+.2f}$")
        st["tg_offset"] = off
    except Exception as e:
        log.warning(f"listen: {e}")

def scan(st):
    d = day_obj(st)
    if time.time() < st.get("stop_until", 0):
        log.info("موقوف مؤقتا (خسائر متتالية)")
        return
    
    lastmap = safe_dict(st, "last_sig")
    pend = safe_list(st, "pending")
    cooldown_symbols = {p.get("sym") for p in pend if p.get("sym")}
    
    if cooldown_symbols:
        log.info(f"cooldown active: {sorted(cooldown_symbols)}")

    for sym in SYMBOLS:
        if sym in cooldown_symbols:
            continue
            
        df = fetch5m(sym)
        if df is None or len(df) < 60:
            log.warning(f"No sufficient data for {sym}")
            continue
        
        rsi, bbu, bbl, sd = indicators(df)
        
        i = len(df) - 1
        ct = df.index[i]
        key = ct.isoformat()
        
        if lastmap.get(sym) == key:
            continue
            
        if np.isnan(sd.iloc[i]):
            continue
            
        entry_loc = ct + timedelta(minutes=5) + timedelta(hours=USER_TZ_OFFSET_H)
        current_hour = entry_loc.hour

        if current_hour < WIN_START_H:
            log.info(f"{sym}: حظر ليلي — تخطي")
            lastmap[sym] = key
            continue

        r = float(rsi.iloc[i])
        cl = float(df["Close"].iloc[i])
        bu = float(bbu.iloc[i])
        bl = float(bbl.iloc[i])
        sdv = float(sd.iloc[i])
        
        dr = None
        band_ref = None
        if r >= RSI_HI and cl >= bu:
            dr = "PUT"; band_ref = bu
        elif r <= RSI_LO and cl <= bl:
            dr = "CALL"; band_ref = bl
            
        if dr is None:
            continue
            
        over = abs(cl - band_ref) / sdv if sdv > 0 else 0.0
        v4 = over >= 0.5
        
        zone, urgency, emoji, urg_text = calc_urgency(cl, band_ref, sdv, dr)
        if urgency == "skip":
            lastmap[sym] = key
            continue
            
        exp = ct + timedelta(minutes=EXPIRY_MIN + 5)
        sent = ct
        arrow = "🔴 PUT (هبوط)" if dr == "PUT" else "🟢 CALL (صعود)"
        sym_copy = fmt_sym(sym)
        
        txt = (f"🎯 *إشارة H2 v5.7.1 (Deriv Live)*\n\n"
               f"📊 *الزوج:* `{sym_copy}`\n"
               f"📈 *الاتجاه:* {arrow}\n"
               f"💰 *السعر الحي الآن:* {fmt_px(cl)}\n\n"
               f"📌 *السبب:* RSI {r:.1f} + السعر "
               f"{'تحت' if dr=='CALL' else 'فوق'} الباند بـ {over:.1f}σ\n"
               f"🏷️ ضمن V4: {'✅' if v4 else '➖'}\n\n"
               f"⌛ الانتهاء: 15 دقيقة (حتى {(ct + timedelta(minutes=EXPIRY_MIN)).strftime('%H:%M')} UTC)\n"
               f"💰 شرط الـ payout: 85% فأعلى فقط\n\n"
               f"📊 تنبيهات اليوم: {d['alerts']}\n"
               f"📝 النتيجة تُسجل تلقائيا عند الانتهاء")
               
        mid_id = tg_send(txt)
        if mid_id is None:
            log.warning(f"فشل إرسال {sym}")
            continue
            
        lastmap[sym] = key
        d["alerts"] += 1
        safe_dict(st, "open")[str(mid_id)] = {
            "sym": sym, "dr": dr, "px": cl, "v4": v4,
            "zone": zone, "urgency": urgency
        }
        safe_list(st, "pending").append({
            "mid": mid_id, "sym": sym, "dr": dr,
            "px": cl, "v4": v4,
            "exp": exp.isoformat(),
            "sent": sent.isoformat(),
        })
        cooldown_symbols.add(sym)
        save_state(st)
        time.sleep(0.5)

def main():
    st = load_state()
    loc = datetime.now(timezone.utc) + timedelta(hours=USER_TZ_OFFSET_H)
    today_local = loc.strftime("%Y-%m-%d")

    last_daily = st.get("last_daily")
    if last_daily is None:
        st["last_daily"] = today_local
    elif last_daily != today_local:
        prev = (loc - timedelta(days=1)).strftime("%Y-%m-%d")
        send_day_summary(st, prev)
        st["last_daily"] = today_local

    d = day_obj(st)
    if st.get("boot_date") != d["date"]:
        st["boot_date"] = d["date"]
        tg_send(f"🚀 بوت H2 بدأ (v5.7.1 Deriv Real-Time)\n"
                f"• المصدر: Deriv API (LIVE)\n"
                f"• أزواج: {len(SYMBOLS)}\n"
                f"• المنطق: RSI(14) + BB(20,2.0)\n"
                f"• الحماية: 3 خسائر متتالية = 4 ساعات توقف")
                
    listen(st)
    resolve_pending(st)
    scan(st)
    st["last_run"] = time.time()
    save_state(st)
    log.info("done")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.exception(f"fatal: {e}")
        try:
            tg_send(f"🚨 *البوت انهار*\nالخطأ: `{e}`\nافحص Actions فورا")
        except Exception:
            pass
        raise
