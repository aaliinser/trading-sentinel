#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — بوت إشارات حي (v5.7.1 Deriv Enhanced)
الاستراتيجية: RSI(14) Wilder + BB(20,2.0) ddof=0 | 5m / 15m
التحديث: إضافة مصدر بيانات Deriv API اللحظي كخيار أول، مع الاحتياط لـ yfinance.
"""
import os, sys, time, json, logging, asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
import requests
import websockets # مكتبة Deriv

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ──
# ملاحظة: رموز Deriv تبدأ عادة بـ frx (مثل frxEURUSD) أو frxUSDJPY
# تأكد من مطابقة الرموز لحسابك في Deriv
SYMBOLS_DERIV = os.getenv("SYMBOLS_DERIV", "frxUSDJPY,frxEURAUD,frxUSDCHF,frxEURCAD,frxCADJPY").split(",")
SYMBOLS_YF = os.getenv("SYMBOLS_H2", "USDJPY=X,EURAUD=X,USDCHF=X,EURCAD=X,CADJPY=X").split(",")

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
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "").strip() # ضع المفتاح هنا في GitHub Secrets وليس في الكود

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("H2v57-Deriv")

# ─── Safe accessors ───
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
                for p in pend:
                    try:
                        exp = datetime.fromisoformat(p["exp"])
                        age_min = (now - exp).total_seconds() / 60.0
                        if age_min <= STATE_CLEANUP_HOURS * 60:
                            fresh.append(p)
                        else:
                            dropped += 1
                    except Exception:
                        dropped += 1
                st["pending"] = fresh
                if dropped:
                    log.info(f"state cleanup: dropped {dropped} stale pending signal(s)")
            return st
        except Exception:
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
    p = {"chat_id": TG_CHAT, "text": text,
         "parse_mode": "Markdown",
         "disable_web_page_preview": True}
    if reply_to:
        p["reply_to_message_id"] = reply_to
    for a in range(3):
        try:
            r = requests.post(url, json=p, timeout=15)
            if r.status_code == 200:
                return r.json().get("result", {}).get("message_id")
            time.sleep(2*a+1)
        except Exception as e:
            log.warning(f"tg {a}: {e}")
    return None

# ─── NEW: Deriv Data Fetcher (Async Wrapper) ───
async def _fetch_deriv_async(symbol, count=220):
    uri = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"
    async with websockets.connect(uri) as ws:
        # Authenticate if token exists
        if DERIV_API_TOKEN:
            await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
            auth_resp = json.loads(await ws.recv())
            if 'error' in auth_resp:
                log.warning(f"Deriv Auth Warning: {auth_resp['error']['message']}")
        
        # Request Candles
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": 300, # 5 minutes
            "count": count,
            "end": "latest"
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
        
        if 'error' in resp:
            raise ValueError(resp['error']['message'])
            
        candles = resp.get('candles', [])
        if not candles:
            return None
            
        df = pd.DataFrame(candles)
        df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
        df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
        df.set_index('epoch', inplace=True)
        df.sort_index(inplace=True)
        return df[['Open','High','Low','Close']].dropna()

def fetch_data_deriv(symbol):
    """Sync wrapper for async Deriv call"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_fetch_deriv_async(symbol))
        loop.close()
        return result
    except Exception as e:
        log.warning(f"Deriv fetch failed for {symbol}: {e}")
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
    يحاول أولاً جلب البيانات من Deriv (لحظي).
    إذا فشل، يعود لـ yfinance (متأخر).
    """
    # 1. Try Deriv First (if mapped correctly)
    # We need to map YF symbols to Deriv symbols roughly or just use the input if it's already Deriv style
    deriv_sym = sym.replace("=X", "").replace("/", "") 
    if not deriv_sym.startswith("frx"):
        deriv_sym = "frx" + deriv_sym
        
    df_deriv = fetch_data_deriv(deriv_sym)
    if df_deriv is not None and len(df_deriv) >= 60:
        return df_deriv
        
    # 2. Fallback to yfinance
    try:
        import yfinance as yf
        for a in range(3):
            try:
                df = yf.Ticker(sym).history(period="3d", interval="5m",
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
                return df
            except Exception as e:
                log.warning(f"{sym} yf fetch {a}: {e}")
                time.sleep(2*a+1)
    except ImportError:
        pass
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
    b = s.replace("=X","").replace("frx","")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

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

# ... (باقي الدوال resolve_pending, send_report, etc. تبقى كما هي في نسختك v5.7.1) ...
# انسخ الأجزاء المتبقية من كودك الأصلي هنا بدون تغيير، فهي تعمل مع أي DataFrame يأتي من fetch5m

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

    # Use SYMBOLS_YF for iteration but fetch_data handles mapping to Deriv internally
    for sym in SYMBOLS_YF:
        if sym in cooldown_symbols:
            continue
            
        df = fetch5m(sym) # This now tries Deriv first!
        if df is None or len(df) < 60:
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

# ... (دوال main و listen وغيرها تبقى كما هي) ...
