#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — بوت إشارات حي (v7.1 Master Hybrid - ZERO SPAM EDITION)
الإصلاح الحاسم: منع تكرار إرسال الملخصات اليومية حتى لو فشل حفظ الحالة (Git Push).
الاستراتيجية: BB(15,2.3) + EMA200 Trend Filter + Storm Filter (ADX/ATR).
الفريم: 5 دقائق / الانتهاء: 15 دقيقة.
التنسيق: مطابق تماماً للإشارة القديمة (RSI removed from text, Sigma kept).
"""
import os, sys, time, json, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ─────────────────────────────────────
SYMBOLS = os.getenv("SYMBOLS_H2", "USDJPY=X,EURAUD=X,USDCHF=X,EURCAD=X,CADJPY=X,GBPUSD=X,USDCAD=X,AUDNZD=X,EURGBP=X").split(",")

# إعدادات المؤشرات للاستراتيجية الهجينة
BB_P = 15          # فترة البولينجر
BB_K = 2.3         # انحراف معياري
EMA_TREND_P = 200  # فلتر الاتجاه طويل المدى
EXPIRY_MIN = 15    # مدة الانتهاء (بالدقائق)
INTERVAL = "5m"    # الفريم الزمني للعمل

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

# Storm Filter Parameters
STORM_ATR_MULT = 1.5      
STORM_ADX_LIMIT = 28.0    
STORM_LOOKBACK = 20       
ATR_PERIOD = 14
ADX_PERIOD = 14

AR_DAYS = ["الاثنين","الثلاثاء","الأربعاء","الخميس",
           "الجمعة","السبت","الأحد"]
AR_MONTHS = ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
             "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("H2v71-ZeroSpam")

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

# ═══════════════════════════════════════════════
# ★ NEW FEATURE: FAST DATA FETCHER USING TICKS ★
# ═══════════════════════════════════════════════
def fetch5m_fast(sym):
    """
    يجلب البيانات بسرعة قصوى عبر بناء شموع 5 دقائق من تيكات لحظية.
    يقلل التأخير إلى الحد الأدنى الممكن مع yfinance.
    """
    ticker = yf.Ticker(sym)
    
    # 1. جلب آخر 3 أيام من الشموع الأساسية (للحسابات التاريخية مثل EMA200)
    hist_df = None
    for a in range(3):
        try:
            hist_df = ticker.history(period="3d", interval=INTERVAL,
                                     auto_adjust=False, actions=False, timeout=20)
            if hist_df is not None and not hist_df.empty:
                break
        except Exception as e:
            log.warning(f"{sym} history fetch {a}: {e}")
            time.sleep(2*a+1)
            
    if hist_df is None or hist_df.empty:
        return None
        
    hist_df = flatten_columns(hist_df)
    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in hist_df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
        
    hist_df = hist_df[required_cols].copy()
    hist_df.index = pd.to_datetime(hist_df.index, utc=True)
    hist_df = hist_df[~hist_df.index.duplicated(keep="last")].sort_index().dropna()
    
    # 2. جلب التيكات اللحظية لآخر ساعة لبناء الشمعة الحالية بدقة عالية
    ticks_df = None
    for a in range(3):
        try:
            # نطلب تيكات لمدة ساعة واحدة فقط لتقليل حجم البيانات وتسريع الجلب
            ticks_df = ticker.history(period="1h", interval="1m", 
                                      auto_adjust=False, actions=False, timeout=15)
            if ticks_df is not None and not ticks_df.empty:
                break
        except Exception as e:
            log.debug(f"{sym} tick fetch {a}: {e}")
            time.sleep(a+1)
            
    if ticks_df is not None and not ticks_df.empty:
        ticks_df = flatten_columns(ticks_df)
        ticks_df.index = pd.to_datetime(ticks_df.index, utc=True)
        
        # تحديد وقت بداية الشمعة الخمسية الحالية
        last_closed_candle_time = hist_df.index[-1]
        current_candle_start = last_closed_candle_time + pd.Timedelta(minutes=5)
        
        # تصفية التيكات التي تقع ضمن الشمعة الحالية
        current_ticks = ticks_df[ticks_df.index >= current_candle_start]
        
        if not current_ticks.empty:
            # بناء الشمعة الحالية من التيكات
            new_candle_data = {
                'Open': current_ticks['Open'].iloc[0],
                'High': current_ticks['High'].max(),
                'Low': current_ticks['Low'].min(),
                'Close': current_ticks['Close'].iloc[-1]
            }
            new_candle_idx = pd.DatetimeIndex([current_candle_start])
            new_candle_df = pd.DataFrame([new_candle_data], index=new_candle_idx)
            
            # دمج الشمعة الجديدة مع التاريخ السابق
            final_df = pd.concat([hist_df, new_candle_df]).sort_index()
        else:
            final_df = hist_df
    else:
        final_df = hist_df
        
    # إزالة الشمعة غير المكتملة إذا كانت موجودة في النهاية
    now = pd.Timestamp.now(tz="UTC")
    if not final_df.empty and final_df.index[-1] + pd.Timedelta(minutes=5) > now:
        final_df = final_df.iloc[:-1]
        
    return final_df

# ─── Core Indicators for v7.0 Hybrid ─────────────────────
def calculate_indicators_v7(df):
    """
    يحسب BB(15, 2.3), EMA(200), ATR, ADX
    """
    c = df["Close"]
    
    # 1. Bollinger Bands (15, 2.3)
    mid = c.rolling(BB_P, min_periods=BB_P).mean()
    sd = c.rolling(BB_P, min_periods=BB_P).std(ddof=0) 
    bu = mid + BB_K * sd
    bl = mid - BB_K * sd
    
    # 2. EMA 200 Trend Filter
    ema200 = c.ewm(span=EMA_TREND_P, adjust=False).mean()
    
    # 3. ATR for Storm Filter
    high = df['High']
    low = df['Low']
    tr1 = high - low
    tr2 = (high - c.shift()).abs()
    tr3 = (low - c.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=ATR_PERIOD).mean()
    
    # 4. ADX for Storm Filter
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)
    
    atr_smooth = true_range.rolling(window=ADX_PERIOD).mean()
    plus_dm_smooth = plus_dm_s.rolling(window=ADX_PERIOD).mean()
    minus_dm_smooth = minus_dm_s.rolling(window=ADX_PERIOD).mean()
    
    atr_safe = atr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / atr_safe)
    minus_di = 100 * (minus_dm_smooth / atr_safe)
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)
    
    di_sum = plus_di + minus_di
    dx_numerator = (plus_di - minus_di).abs()
    dx_denominator = di_sum.replace(0, np.nan)
    dx = 100 * (dx_numerator / dx_denominator)
    dx = dx.fillna(0)
    adx = dx.rolling(window=ADX_PERIOD).mean()
    
    return bl, bu, ema200, atr, adx, sd

def fmt_sym(s):
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def fmt_px(v):
    v = float(v)
    return f"{v:.3f}" if v > 50 else f"{v:.5f}"

def calc_urgency(cl, band, sdv, direction):
    if sdv <= 0.0001:
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
        except Exception as e:
            log.error(f"Pending parse error for {p.get('sym')}: {e}")
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

        df = fetch5m_fast(p["sym"]) # Use fast version here too
        target = exp - timedelta(minutes=5)
        exit_px = None

        if df is not None and not df.empty:
            if target in df.index:
                exit_px = float(df.loc[target, "Close"])
            else:
                earlier = df[df.index <= target]
                if not earlier.empty:
                    exit_px = float(earlier["Close"].iloc[-1])
                    log.info(f"{p['sym']}: target {target} missing, using fallback at {earlier.index[-1]}")

        if exit_px is None:
            if age_min > PENDING_SOFT_TIMEOUT_MIN:
                log.warning(f"void pending {p['sym']} (no data after {age_min:.0f}m)")
                done.append(p)
                voided_count += 1
            continue

        entry = float(p["px"])
        if abs(exit_px - entry) < 0.00001:
            log.info(f"tie void {p['sym']}")
            done.append(p)
            voided_count += 1
            continue

        win = exit_px > entry if p["dr"] == "CALL" else exit_px < entry
        sent = datetime.fromisoformat(p["sent"])
        loc = sent + timedelta(hours=USER_TZ_OFFSET_H)

        sim_list = safe_list(st, "sim")
        new_record = {
            "dl": loc.strftime("%Y-%m-%d"),
            "hl": loc.hour,
            "sym": p["sym"], "dr": p["dr"],
            "v4": bool(p.get("v4", False)), 
            "win": bool(win),
            "px": entry, "ex": exit_px,
        }
        sim_list.append(new_record)
        save_state(st) 

        if len(sim_list) > 2000:
            st["sim"] = sim_list[-2000:]
            save_state(st)

        log.info(f"auto-resolved {p['sym']} {p['dr']} win={win} | Entry:{entry:.5f} Exit:{exit_px:.5f}")
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
        log.info(f"resolve_pending summary: resolved={resolved_count} voided={voided_count} forced={forced_count} remaining={len(pend)}")

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
    txt = (f"📊 *تقرير الأداء (v7.1 Fast)*\n\n"
           f"*إشارات:* *{n}* | فوز *{wr}%*\n\n"
           f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def send_day_summary(st, day_str):
    sim = [x for x in safe_list(st, "sim") if x["dl"] == day_str]
    win_sig = [x for x in sim if x["hl"] >= WIN_START_H]
    out_sig = [x for x in sim if x["hl"] < WIN_START_H]
    man = [x for x in safe_list(st, "log") if x.get("date") == day_str]
    n = len(win_sig)
    w = sum(1 for x in win_sig if x["win"])
    l = n - w
    wr = round(100*w/n, 1) if n else 0.0
    mn = len(man)
    mw = sum(1 for x in man if x["win"])
    mwr = round(100*mw/mn, 1) if mn else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    pnl = round(w * win_payout - l * stake, 2)
    txt = (f"🌙 *ملخص يوم {ar_day(day_str)} {day_str}*\n"
           f"(نافذة العد: {WIN_START_H}:00 – 24:00 بتوقيتك)\n\n"
           f"• إشارات النافذة: *{n}*\n"
           f"• تلقائي: فوز {w} / خسارة {l} | *{wr}%*\n"
           f"• صافي تقديري: {pnl:+.2f}$\n"
           f"• إشارات خارج النافذة: {len(out_sig)}\n"
           f"• ردودك اليدوية: {mn} | {mwr}%\n\n"
           f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def send_week_summary(st, days):
    sim = [x for x in safe_list(st, "sim") if x["dl"] in days]
    if not sim:
        return
    n = len(sim)
    w = sum(1 for x in sim if x["win"])
    l = n - w
    wr = 100*w/n
    p_star = 1.0/1.9
    se = (p_star*(1-p_star)/n) ** 0.5
    z = (wr/100 - p_star)/se if se > 0 else 0.0
    man = [x for x in safe_list(st, "log") if x.get("date") in days]
    mn = len(man)
    mw = sum(1 for x in man if x["win"])
    mwr = round(100*mw/mn, 1) if mn else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    txt = (f"📅 *ملخص أسبوع التداول (v7.1 Fast)*\n"
           f"({ar_day(days[0])} {days[0]} → "
           f"{ar_day(days[-1])} {days[-1]})\n\n"
           f"📆 *تفصيل الأيام:*\n")
    for dstr in days:
        ds = [x for x in sim if x["dl"] == dstr]
        dn = len(ds)
        dw = sum(1 for x in ds if x["win"])
        dwr = round(100*dw/dn, 1) if dn else 0.0
        txt += f"• {ar_day(dstr)} {dstr}: {dn} إشارة | {dwr}%\n"

    buckets = [
        ("صباحاَ (09-14)", lambda h: 9 <= h < 14),
        ("عصراَ (14-19)", lambda h: 14 <= h < 19),
        ("مساءً (19-24)", lambda h: h >= 19),
    ]
    txt += f"\n⏰ *تفصيل أوقات اليوم:*\n"
    bucket_stats = []
    for name, cond in buckets:
        bs = [x for x in sim if cond(int(x.get("hl", -1)))]
        bn = len(bs)
        bw = sum(1 for x in bs if x["win"])
        bwr = round(100*bw/bn, 1) if bn else 0.0
        bucket_stats.append((name, bn, bwr))
        txt += f"• {name}: {bn} إشارة | {bwr}%\n"
    active = [b for b in bucket_stats if b[1] > 0]
    if active:
        best_b = max(active, key=lambda b: b[2])
        txt += f"🏆 أفضل وقت: {best_b[0]} ({best_b[2]}%)\n"

    txt += f"\n🧩 *تفصيل الأزواج:*\n"
    pair_stats = []
    for sym in sorted({x["sym"] for x in sim}):
        ss = [x for x in sim if x["sym"] == sym]
        sn = len(ss)
        sw = sum(1 for x in ss if x["win"])
        swr = round(100*sw/sn, 1) if sn else 0.0
        pair_stats.append((sym, sn, swr, sn - sw))
        txt += f"• {fmt_sym(sym)}: {sn} | {swr}%\n"
    if pair_stats:
        best_p = max(pair_stats, key=lambda p: p[2])
        txt += f"🏆 أفضل زوج: {fmt_sym(best_p[0])} ({best_p[2]}%)\n"

    max_streak = 0
    cur = 0
    for x in sim:
        if not x["win"]:
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    txt += f"\n🔁 *تحليل أنماط الخسارة:*\n"
    txt += f"• أطول سلسلة خسائر متتالية: {max_streak}\n"
    if pair_stats:
        worst_p = max(pair_stats, key=lambda p: p[3])
        if worst_p[3] > 0:
            txt += f"• أكثر الأزواج خسارة: {fmt_sym(worst_p[0])} ({worst_p[3]} خسائر)\n"
    if active:
        worst_b = min(active, key=lambda b: b[2])
        txt += f"• أضعف وقت: {worst_b[0]} ({worst_b[2]}%)\n"

    pnl = round(w * win_payout - l * stake, 2)
    txt += (f"\n• إجمالي الإشارات: *{n}*\n"
            f"• تلقائي: فوز {w} / خسارة {l} | *{round(wr,1)}%*\n"
            f"• صافي تقديري: {pnl:+.2f}$\n"
            f"• Z-score: {z:+.2f} "
            f"{'✅ دلالة حقيقية' if z > 1.96 else '⚠️ ضمن الضجيج'}\n"
            f"• ردودك اليدوية: {mn} | {mwr}%\n\n"
            f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def send_month_summary(st, y, m):
    prefix = f"{y:04d}-{m:02d}"
    sim = [x for x in safe_list(st, "sim")
           if x["dl"].startswith(prefix) and x["dl"] >= MONTH_START]
    if not sim:
        tg_send(f"🗓️ *ملخص شهر {ar_month(m)} {y}*\nلا إشارات مسجلة")
        return
    n = len(sim)
    w = sum(1 for x in sim if x["win"])
    l = n - w
    wr = 100*w/n
    p_star = 1.0/1.9
    se = (p_star*(1-p_star)/n) ** 0.5
    z = (wr/100 - p_star)/se if se > 0 else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    pnl = round(w * win_payout - l * stake, 2)
    txt = (f"🗓️ *ملخص شهر {ar_month(m)} {y}*\n"
           f"• إجمالي الإشارات: *{n}*\n"
           f"• فوز {w} / خسارة {l} | *{round(wr,1)}%*\n"
           f"• صافي تقديري: {pnl:+.2f}$\n"
           f"• Z-score: {z:+.2f}\n\n"
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
            
        # ★ USE THE NEW FAST FETCHER HERE ★
        df = fetch5m_fast(sym)
        if df is None or len(df) < 220: 
            continue
        
        bl, bu, ema200, atr, adx, sd = calculate_indicators_v7(df)
        
        i = len(df) - 1
        ct = df.index[i]
        key = ct.isoformat()
        
        if lastmap.get(sym) == key:
            continue
            
        if pd.isna(bl.iloc[i]) or pd.isna(bu.iloc[i]) or pd.isna(ema200.iloc[i]) or \
           pd.isna(atr.iloc[i]) or pd.isna(adx.iloc[i]) or pd.isna(sd.iloc[i]):
            continue
            
        entry_loc = ct + timedelta(minutes=5) + timedelta(hours=USER_TZ_OFFSET_H)
        current_hour = entry_loc.hour

        if current_hour < WIN_START_H:
            log.info(f"{sym}: حظر ليلي — تخطي")
            lastmap[sym] = key
            continue

        cl = float(df["Close"].iloc[i])
        bl_val = float(bl.iloc[i])
        bu_val = float(bu.iloc[i])
        ema_val = float(ema200.iloc[i])
        current_atr = float(atr.iloc[i])
        current_adx = float(adx.iloc[i])
        sdv = float(sd.iloc[i])
        
        # ─── STORM FILTER CHECK ───
        storm_detected = False
        reason = ""
        
        lookback_slice = atr.iloc[max(0, i-STORM_LOOKBACK):i] 
        valid_lookback = lookback_slice.dropna()
        
        if len(valid_lookback) >= 5:
            baseline_atr = float(valid_lookback.mean())
        else:
            baseline_atr = current_atr 
            
        if not pd.isna(current_atr) and not pd.isna(baseline_atr):
             if baseline_atr > 0.0001 and current_atr > (baseline_atr * STORM_ATR_MULT):
                storm_detected = True
                reason = f"ATR Spike ({current_atr:.5f} vs Base {baseline_atr:.5f})"
                
        elif not pd.isna(current_adx) and current_adx > STORM_ADX_LIMIT:
            storm_detected = True
            reason = f"Strong Trend (ADX={current_adx:.1f} > {STORM_ADX_LIMIT})"
            
        if storm_detected:
            log.warning(f"🌪️ STORM FILTER ACTIVATED FOR {sym}: {reason}. Signal SKIPPED.")
            lastmap[sym] = key 
            continue

        # ─── CORE LOGIC: BB + EMA200 TREND FILTER ───
        dr = None
        band_ref = None
        
        if cl <= bl_val and cl >= ema_val:
            dr = "CALL"
            band_ref = bl_val
            
        elif cl >= bu_val and cl <= ema_val:
            dr = "PUT"
            band_ref = bu_val
            
        if dr is None:
            continue
            
        over = 0.0
        v4 = False
        if sdv > 0.0001:
            over = abs(cl - band_ref) / sdv
            v4 = over >= 0.5 
        
        zone, urgency, emoji, urg_text = calc_urgency(cl, band_ref, sdv, dr)
        if urgency == "skip":
            lastmap[sym] = key
            continue
        
        if dr == "CALL":
            strong_entry = bl_val
            mid_entry = (cl + bl_val) / 2
        else:
            strong_entry = bu_val
            mid_entry = (cl + bu_val) / 2
            
        exp = ct + timedelta(minutes=EXPIRY_MIN + 5)
        sent = ct
        arrow = "🔴 PUT (هبوط)" if dr == "PUT" else "🟢 CALL (صعود)"
        sym_copy = fmt_sym(sym)
        
        txt = (f"🎯 *إشارة H2 v7.1 Fast — BB + Trend*\n\n"
               f"📊 *الزوج:* `{sym_copy}`\n"
               f"📈 *الاتجاه:* {arrow}\n"
               f"💰 *السعر الحي الآن:* {fmt_px(cl)}\n\n"
               f"🎯 *مناطق الدخول:*\n"
               f"• قوية (عند الباند): {fmt_px(strong_entry)}\n"
               f"• وسطى (مقبولة): {fmt_px(mid_entry)}\n\n"
               f"⏱️ *التعليمات:*\n"
               f"{emoji} {urg_text}\n\n"
               f"📌 *السبب:* السعر {'تحت' if dr=='CALL' else 'فوق'} الباند بـ {over:.1f}σ\n"
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

    # ─── FIX: Robust Daily Summary Logic (Prevent Spamming Even on Git Failures) ───
    last_daily = st.get("last_daily")
    
    # We only send summary if:
    # 1. There is a previous day recorded.
    # 2. That previous day is NOT the current local day.
    # This ensures we never summarize "today" while it's still ongoing.
    if last_daily is not None and last_daily != today_local:
        prev_day_str = last_daily 
        send_day_summary(st, prev_day_str)
        
        # CRITICAL UPDATE: Mark that we have handled the transition to TODAY
        st["last_daily"] = today_local
        
        # FORCE SAVE HERE to prevent re-triggering in next run within same minute
        save_state(st) 
        log.info(f"✅ Daily summary sent for {prev_day_str}. State updated to {today_local}.")

    elif last_daily is None:
        # First ever run, just set the date without sending summary
        st["last_daily"] = today_local
        save_state(st)

    # ─── Monthly & Weekly Logic (Keep as is but ensure saves happen) ───
    if st.get("last_month") is None:
        st["last_month"] = loc.strftime("%Y-%m")
        save_state(st)
        
    if loc.day == 1 and st.get("last_month") != loc.strftime("%Y-%m"):
        if loc.month > 1:
            py, pm = loc.year, loc.month - 1
        else:
            py, pm = loc.year - 1, 12
        send_month_summary(st, py, pm)
        st["last_month"] = loc.strftime("%Y-%m")
        save_state(st)

    now_utc = datetime.now(timezone.utc)
    iso = loc.isocalendar()
    wk = f"{iso[0]}-W{iso[1]}"
    
    if now_utc.weekday() == 5 and st.get("last_week") != wk:
        monday = loc - timedelta(days=loc.weekday())
        days = [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
        send_week_summary(st, days)
        st["last_week"] = wk
        save_state(st)

    d = day_obj(st)
    if st.get("boot_date") != d["date"]:
        st["boot_date"] = d["date"]
        tg_send(f"🚀 بوت H2 بدأ (v7.1 Master Hybrid - ZERO SPAM)\n"
                f"• أزواج: {len(SYMBOLS)} (تم التوسع)\n"
                f"• المنطق: BB(15,2.3) + EMA200 Filter\n"
                f"• الحماية: Storm Filter (ADX/ATR) نشط\n"
                f"• الفريم: 5 دقائق / الانتهاء: 15 دقيقة\n"
                f"• ⚡ الوضع السريع: يعتمد على بناء شموع ديناميكية من التيكات\n"
                f"• 🔑 لا يحتاج لمفتاح Deriv\n"
                f"• ✨ بدون سقف تنبيهات — لن تضيع إشارة\n"
                f"• ❄️ تبريد لكل زوج: إشارة واحدة حتى حسم النتيجة\n"
                f"• 🌙 حظر ليلي: لا إشارات من 12 إلى 9 صباحا\n"
                f"• ⏳ حماية ضد التجميد: timeout {PENDING_HARD_TIMEOUT_MIN}د\n"
                f"• 🤖 تسجيل تلقائي + رد تلقائي بالنتيجة\n"
                f"• 📊 ملخص يومي عند منتصف الليل\n"
                f"• 📅 ملخص أسبوعي السبت (أيام + أوقات + أزواج + أنماط خسارة)\n"
                f"• 🗓️ ملخص شهري أول كل شهر\n"
                f"• 🛡️ الحماية: 3 خسائر متتالية = 4 ساعات توقف")
                
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
