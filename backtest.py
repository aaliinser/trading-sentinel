#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
غيث — مصفوفة v29: اختبار كل التركيبات + الصلابة
=====================================================================
يفحص:
1) الاتجاه: استمرار (v28) ← مقابل → عكس (Fade)
2) الانتهاء: 15 دقيقة ← مقابل → 30 دقيقة
3) نقطة الدخول: إغلاق التأكيد ← مقابل → إغلاق الرفض
4) اختبار الصلابة: نصفان زمنيان (أول 30 يوم + آخر 30 يوم)

يختار التركيبات الرابحة والصلبة فقط.
=====================================================================
"""
import os, sys, time, json, logging
from datetime import datetime, timezone
import numpy as np, pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# ============ الإعدادات ============
SYMBOLS = [
    "USDJPY=X","AUDJPY=X","EURJPY=X","EURUSD=X","GBPUSD=X",
    "EURGBP=X","CADJPY=X","EURCAD=X","GBPCAD=X","AUDCHF=X",
    "AUDUSD=X","USDCHF=X","CHFJPY=X","AUDCAD=X","USDCAD=X",
    "EURAUD=X","EURCHF=X","GBPJPY=X","GBPCHF=X","GBPAUD=X"
]

SCAN_TF = "15m"
SNIPER_TF = "5m"
TREND_TF = "1h"
EXPIRY_MIN = 15
LVL_LB = 60
EMA_F = 35
EMA_S = 50
RSI_P = 14
ATR_P = 14
LVL_PROX = 1.0
MAX_DIST_EMA = 3.0
MAX_DEV = 0.0020
MAX_AHEAD = 0.0008
TOUCH_TOL = 0.0005
REJ_BODY = 0.30
WICK_BODY = 2.0
IMPULSE_ATR = 2.5
CONFL_ATR = 0.3
RN_LARGE = 0.5
RN_SMALL = 0.005
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

# ============ Logging ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Backtest_v29")

# ============ الجلب ============
def fetch(sym, iv, period="7d"):
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(sym).history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for c in ["Open","High","Low","Close","Volume"]:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[["Open","High","Low","Close","Volume"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(subset=["Open","High","Low","Close"], inplace=True)
            df = df[(df["High"]>=df["Low"]) & (df["Open"]>0) & (df["Close"]>0)]
            if not df.empty:
                return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return None

# ============ المؤشرات ============
def add_indicators(df):
    if df is None or df.empty or len(df) < 200:
        return df
    df = df.copy()
    if HAS_TA:
        df["EMA_35"] = ta.ema(df["Close"], length=EMA_F)
        df["EMA_50"] = ta.ema(df["Close"], length=EMA_S)
        df["RSI"] = ta.rsi(df["Close"], length=RSI_P)
        a = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_P)
        if a is not None:
            df["ATR"] = a
    else:
        df["EMA_35"] = df["Close"].ewm(span=EMA_F, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=EMA_S, adjust=False).mean()
        d = df["Close"].diff()
        g = d.clip(lower=0)
        l = -d.clip(upper=0)
        ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
        al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
        df["RSI"] = (100 - (100 / (1 + ag / al.replace(0, np.nan)))).fillna(50)
        pc = df["Close"].shift(1)
        tr = pd.concat([
            df["High"]-df["Low"],
            (df["High"]-pc).abs(),
            (df["Low"]-pc).abs()
        ], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1/ATR_P, min_periods=ATR_P).mean()
    df["RS"] = df["Low"].rolling(LVL_LB, min_periods=20).min()
    df["RR"] = df["High"].rolling(LVL_LB, min_periods=20).max()
    df["H20"] = df["High"].rolling(20, min_periods=10).max()
    df["L20"] = df["Low"].rolling(20, min_periods=10).min()
    df["UWICK"] = df["High"] - df[["Open","Close"]].max(axis=1)
    df["LWICK"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    return df

# ============ قواعد v28 (للعثور على لحظات الإرهاق) ============
def trend_aligned(last_15, prev_15, last_h1):
    if any(pd.isna(v) for v in [last_15["EMA_35"], last_15["EMA_50"],
                                 prev_15["EMA_35"], last_h1["EMA_35"],
                                 last_h1["EMA_50"], last_15["Close"], last_h1["Close"]]):
        return None
    hb = last_h1["Close"] > last_h1["EMA_35"] > last_h1["EMA_50"]
    hr = last_h1["Close"] < last_h1["EMA_35"] < last_h1["EMA_50"]
    mb = last_15["Close"] > last_15["EMA_35"] > last_15["EMA_50"] and last_15["EMA_35"] > prev_15["EMA_35"]
    mr = last_15["Close"] < last_15["EMA_35"] < last_15["EMA_50"] and last_15["EMA_35"] < prev_15["EMA_35"]
    if hb and mb: return "CALL"
    if hr and mr: return "PUT"
    return None

def find_level(last, dr):
    c = float(last["Close"])
    a = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
    if a <= 0: return None
    md = LVL_PROX * a
    cand = []
    if dr == "CALL" and pd.notna(last.get("RS")):
        sp = float(last["RS"])
        if abs(c - sp) <= md:
            cand.append((sp, "SUPPORT"))
    if dr == "PUT" and pd.notna(last.get("RR")):
        r = float(last["RR"])
        if abs(c - r) <= md:
            cand.append((r, "RESISTANCE"))
    step = RN_LARGE if c > 50 else RN_SMALL
    if step > 0:
        nr = round(c / step) * step
        if abs(c - nr) <= md:
            cand.append((nr, "ROUND_NUMBER"))
    if not cand:
        return None
    cand.sort(key=lambda x: abs(c - x[0]))
    return cand[0][0]

def ema_distance_ok(last):
    c = float(last["Close"])
    e = float(last["EMA_35"])
    a = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
    return a > 0 and abs(c - e) <= MAX_DIST_EMA * a

def touch_rej_confirm(d5_view, level, dr):
    """يرجع: (rej_ok, conf_ok, rej_close, conf_close)"""
    if len(d5_view) < 3:
        return False, False, None, None
    conf, rej, prev = d5_view.iloc[-1], d5_view.iloc[-2], d5_view.iloc[-3]
    rej_close = float(rej["Close"])
    conf_close = float(conf["Close"])
    rej_body = abs(float(rej["Close"]) - float(rej["Open"]))
    rej_range = float(rej["High"]) - float(rej["Low"])
    if rej_range <= 0 or rej_body <= 0:
        return False, False, None, None
    br = rej_body / rej_range
    brej = br >= REJ_BODY
    t = TOUCH_TOL * float(rej["Close"])
    if dr == "CALL":
        touched = float(rej["Low"]) <= level + t
    else:
        touched = float(rej["High"]) >= level - t
    if not touched:
        return False, False, None, None
    if dr == "CALL":
        lw = float(rej.get("LWICK", 0)) if pd.notna(rej.get("LWICK")) else 0
        pin = lw >= WICK_BODY * rej_body
        eng = rej["Close"] > rej["Open"] and prev["Close"] < prev["Open"] and rej["Close"] >= prev["Open"] and rej["Open"] <= prev["Close"]
        rej_ok = (brej or pin or eng) and rej_close > level
        conf_ok = conf_close > rej_close
    else:
        uw = float(rej.get("UWICK", 0)) if pd.notna(rej.get("UWICK")) else 0
        pin = uw >= WICK_BODY * rej_body
        eng = rej["Close"] < rej["Open"] and prev["Close"] > prev["Open"] and rej["Close"] <= prev["Open"] and rej["Open"] >= prev["Close"]
        rej_ok = (brej or pin or eng) and rej_close < level
        conf_ok = conf_close < rej_close
    return rej_ok, conf_ok, rej_close, conf_close

def impulse_ok(d5_view, dr):
    if len(d5_view) < 5:
        return True
    last3 = d5_view.iloc[-4:-1]
    net = float((last3["Close"] - last3["Open"]).sum())
    atr = float(d5_view.iloc[-1]["ATR"]) if pd.notna(d5_view.iloc[-1].get("ATR")) else 0
    if atr <= 0:
        return True
    if dr == "PUT" and net > IMPULSE_ATR * atr:
        return False
    if dr == "CALL" and net < -IMPULSE_ATR * atr:
        return False
    return True

def deviation_ok(level, entry_price, dr):
    dn = (level - entry_price) / entry_price
    up = (entry_price - level) / entry_price
    if dr == "PUT":
        if dn > MAX_DEV: return False
        if up > MAX_AHEAD: return False
    else:
        if up > MAX_DEV: return False
        if dn > MAX_AHEAD: return False
    return True

# ============ جمع الصفقات الخام (بوابات v28) ============
def collect_raw_trades(sym):
    """يجمع كل لحظة استوفت بوابات v28 - بدون قرار اتجاه/انتهاء"""
    log.info(f"=== {sym} ===")
    d15 = fetch(sym, SCAN_TF, f"{HISTORY_DAYS}d")
    d5  = fetch(sym, SNIPER_TF, "60d")
    d1h = fetch(sym, TREND_TF, f"{HISTORY_DAYS}d")
    if d15 is None or d5 is None or d1h is None:
        log.warning(f"{sym}: فشل جلب البيانات")
        return []
    d15 = add_indicators(d15)
    d5  = add_indicators(d5)
    d1h = add_indicators(d1h)
    if d15 is None or d5 is None or d1h is None:
        return []
    if len(d15) < 300 or len(d5) < 1000 or len(d1h) < 100:
        log.warning(f"{sym}: بيانات غير كافية")
        return []

    raw = []
    for i in range(10, len(d5)):
        cur_time = d5.index[i]
        m15 = d15[d15.index <= cur_time]
        m1h = d1h[d1h.index <= cur_time]
        if len(m15) < 5 or len(m1h) < 3:
            continue
        last_15 = m15.iloc[-1]
        prev_15 = m15.iloc[-2]
        last_h1 = m1h.iloc[-1]
        dr = trend_aligned(last_15, prev_15, last_h1)
        if dr is None:
            continue
        if not ema_distance_ok(last_15):
            continue
        level = find_level(last_15, dr)
        if level is None:
            continue
        d5_view = d5.iloc[max(0, i-20):i+1]
        if len(d5_view) < 10:
            continue
        rej_ok, conf_ok, rej_close, conf_close = touch_rej_confirm(d5_view, level, dr)
        if not (rej_ok and conf_ok):
            continue
        if not impulse_ok(d5_view, dr):
            continue
        entry_conf = float(d5_view.iloc[-1]["Close"])
        if not deviation_ok(level, entry_conf, dr):
            continue
        # أسعار الخروج المحتملة
        end_idx_15 = i + 3   # 15 دقيقة = 3 شموع 5m
        end_idx_30 = i + 6   # 30 دقيقة = 6 شموع 5m
        exit_15 = float(d5.iloc[end_idx_15]["Close"]) if end_idx_15 < len(d5) else None
        exit_30 = float(d5.iloc[end_idx_30]["Close"]) if end_idx_30 < len(d5) else None
        raw.append({
            "time": cur_time,
            "dr_detected": dr,     # الاتجاه الذي اكتشفه v28
            "level": level,
            "rej_close": rej_close,
            "conf_close": conf_close,
            "exit_15": exit_15,
            "exit_30": exit_30,
        })
    log.info(f"{sym}: {len(raw)} لحظة v28")
    return raw

# ============ تقييم تركيبة معينة ============
def evaluate_combo(raw_trades, mode, expiry, entry_at):
    """
    mode: "continue" (استمرار v28) | "fade" (عكس v28)
    expiry: 15 | 30
    entry_at: "confirm" (إغلاق التأكيد) | "reject" (إغلاق الرفض)
    """
    wins = 0
    losses = 0
    trades = []
    for t in raw_trades:
        if expiry == 15 and t["exit_15"] is None:
            continue
        if expiry == 30 and t["exit_30"] is None:
            continue
        exit_price = t["exit_15"] if expiry == 15 else t["exit_30"]
        entry_price = t["conf_close"] if entry_at == "confirm" else t["rej_close"]
        dr_detected = t["dr_detected"]
        # قرار الدخول الفعلي
        if mode == "continue":
            dr_enter = dr_detected
        else:  # fade
            dr_enter = "PUT" if dr_detected == "CALL" else "CALL"
        win = (exit_price > entry_price) if dr_enter == "CALL" else (exit_price < entry_price)
        if win:
            wins += 1
        else:
            losses += 1
        trades.append({
            "time": t["time"],
            "win": win,
            "dr_enter": dr_enter,
            "entry": entry_price,
            "exit": exit_price,
        })
    total = wins + losses
    wr = round(100 * wins / total, 2) if total > 0 else 0.0
    pnl = round(wins * STAKE * PAYOUT - losses * STAKE, 2)
    return {"total": total, "wins": wins, "losses": losses, "wr": wr, "pnl": pnl, "trades": trades}

# ============ اختبار الصلابة (نصفان زمنيان) ============
def robustness_check(trades):
    if len(trades) < 20:
        return False, 0, 0
    mid_time = trades[len(trades)//2]["time"]
    first_half = [t for t in trades if t["time"] < mid_time]
    second_half = [t for t in trades if t["time"] >= mid_time]
    if len(first_half) < 10 or len(second_half) < 10:
        return False, 0, 0
    wr1 = 100 * sum(1 for t in first_half if t["win"]) / len(first_half)
    wr2 = 100 * sum(1 for t in second_half if t["win"]) / len(second_half)
    robust = wr1 >= 52.6 and wr2 >= 52.6
    return robust, round(wr1, 1), round(wr2, 1)

# ============ تحليل الجلسات للتركيبة الفائزة ============
def session_analysis(trades):
    sessions = {}
    for t in trades:
        h = t["time"].hour
        if 2 <= h < 6: s = "آسيا"
        elif 7 <= h < 11: s = "لندن-افتتاح"
        elif 11 <= h < 15: s = "لندن"
        elif 15 <= h < 20: s = "نيويورك"
        else: s = "هادئة"
        if s not in sessions:
            sessions[s] = {"w": 0, "t": 0}
        sessions[s]["t"] += 1
        if t["win"]:
            sessions[s]["w"] += 1
    return sessions

# ============ بناء التقرير ============
def fmt_sym(s):
    b = s.replace("=X", "")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def build_matrix_report():
    log.info(f"🚀 بدء مصفوفة v29: {len(SYMBOLS)} زوجاً × {HISTORY_DAYS} يوم")
    start = time.time()

    # 1) جمع الصفقات الخام لكل زوج
    all_raw = {}
    for sym in SYMBOLS:
        try:
            all_raw[sym] = collect_raw_trades(sym)
            time.sleep(1)
        except Exception as e:
            log.error(f"{sym}: {e}")
            all_raw[sym] = []

    # 2) المصفوفة: 2 اتجاه × 2 انتهاء × 2 دخول = 8 تركيبات
    combos = [
        ("continue", 15, "confirm", "استمرار v28 | 15د | تأكيد"),
        ("continue", 30, "confirm", "استمرار v28 | 30د | تأكيد"),
        ("continue", 15, "reject",  "استمرار v28 | 15د | رفض"),
        ("continue", 30, "reject",  "استمرار v28 | 30د | رفض"),
        ("fade",     15, "confirm", "عكس (Fade) | 15د | تأكيد"),
        ("fade",     30, "confirm", "عكس (Fade) | 30د | تأكيد"),
        ("fade",     15, "reject",  "عكس (Fade) | 15د | رفض"),
        ("fade",     30, "reject",  "عكس (Fade) | 30د | رفض"),
    ]

    results = []
    for mode, expiry, entry_at, label in combos:
        total_t = 0
        total_w = 0
        total_pnl = 0.0
        by_sym = {}
        all_trades_for_robustness = []
        for sym, raw in all_raw.items():
            if not raw:
                continue
            r = evaluate_combo(raw, mode, expiry, entry_at)
            total_t += r["total"]
            total_w += r["wins"]
            total_pnl += r["pnl"]
            by_sym[sym] = r
            all_trades_for_robustness.extend(r["trades"])
        wr = round(100 * total_w / total_t, 2) if total_t > 0 else 0
        robust, wr1, wr2 = robustness_check(all_trades_for_robustness)
        sessions = session_analysis(all_trades_for_robustness) if all_trades_for_robustness else {}
        # أفضل/أسوأ 3 أزواج
        ranked = sorted(by_sym.items(), key=lambda x: x[1]["wr"] if x[1]["total"] >= 10 else -1, reverse=True)
        top3 = [(s, d) for s, d in ranked[:3] if d["total"] >= 10]
        bottom3 = [(s, d) for s, d in ranked[-3:] if d["total"] >= 10]
        # أفضل جلسة
        best_session = max(sessions.items(), key=lambda x: (100*x[1]["w"]/x[1]["t"] if x[1]["t"] else 0)) if sessions else None
        results.append({
            "label": label,
            "mode": mode,
            "expiry": expiry,
            "entry_at": entry_at,
            "total": total_t,
            "wins": total_w,
            "wr": wr,
            "pnl": round(total_pnl, 2),
            "robust": robust,
            "wr_h1": wr1,
            "wr_h2": wr2,
            "top3": top3,
            "bottom3": bottom3,
            "best_session": best_session,
            "sessions": sessions,
        })

    # 3) بناء الرسالة
    winners = [r for r in results if r["robust"] and r["wr"] >= 55]
    marginal = [r for r in results if r["robust"] and 52.6 <= r["wr"] < 55]
    losers = [r for r in results if not r["robust"] or r["wr"] < 52.6]

    msg = f"🎯 *مصفوفة v29 — تقرير شامل*\n"
    msg += f"(20 زوجاً × 60 يوم × 8 تركيبات)\n\n"

    msg += f"📋 *الجدول الكامل:*\n"
    msg += f"```\n"
    msg += f"{'التركيبة':<32} {'الصفقات':>7} {'الفوز':>6} {'الربح':>9} {'صلب؟':>5}\n"
    msg += f"{'-'*32} {'-'*7} {'-'*6} {'-'*9} {'-'*5}\n"
    for r in sorted(results, key=lambda x: x["wr"], reverse=True):
        robust_mark = "✅" if r["robust"] else "❌"
        pnl_mark = f"{r['pnl']:+.0f}$"
        msg += f"{r['label']:<32} {r['total']:>7} {r['wr']:>5.1f}% {pnl_mark:>9} {robust_mark:>5}\n"
    msg += f"```\n\n"

    if winners:
        msg += f"🏆 *التركيبات الرابحة الصلبة ({len(winners)}):*\n"
        for r in winners:
            msg += f"\n*{r['label']}*\n"
            msg += f"• نسبة الفوز: *{r['wr']}%*\n"
            msg += f"• الربح 60 يوم: *{r['pnl']:+.2f}$*\n"
            msg += f"• الصلابة: النصف الأول {r['wr_h1']}% | الثاني {r['wr_h2']}%\n"
            if r["top3"]:
                top_txt = " | ".join(f"*{fmt_sym(s)}* {d['wr']}%" for s, d in r["top3"])
                msg += f"• أفضل 3 أزواج: {top_txt}\n"
            if r["best_session"] and r["best_session"][1]["t"] >= 20:
                s_name = r["best_session"][0]
                s_wr = round(100*r["best_session"][1]["w"]/r["best_session"][1]["t"], 1)
                msg += f"• أفضل جلسة: *{s_name}* ({s_wr}%)\n"
    else:
        msg += f"❌ *لا توجد تركيبة رابحة صلبة!* (فوق 55% في النصفين)\n\n"

    if marginal:
        msg += f"\n⚠️ *تركيبات هامشية (فوق التعادل لكن <55%):*\n"
        for r in marginal:
            msg += f"• {r['label']}: {r['wr']}% ({r['pnl']:+.0f}$)\n"

    msg += f"\n💡 *الخلاصة:*\n"
    if winners:
        best = max(winners, key=lambda x: (x["wr"], x["pnl"]))
        msg += f"🥇 *أفضل تركيبة:* {best['label']}\n"
        msg += f"   نسبة الفوز: *{best['wr']}%* | ربح 60 يوم: *{best['pnl']:+.0f}$*\n"
        if best["top3"]:
            top_txt = ", ".join(f"*{fmt_sym(s)}* ({d['wr']}%)" for s, d in best["top3"])
            msg += f"🎯 *الأزواج المختارة لـ v29:* {top_txt}\n"
        if best["best_session"] and best["best_session"][1]["t"] >= 20:
            s_name = best["best_session"][0]
            msg += f"⏰ *أفضل جلسة:* {s_name}\n"
        msg += f"\n✅ *توصية:* نعتمد هذه التركيبة كـ v29 وندمجها في البوت اليومي\n"
    elif marginal:
        msg += f"⚠️ لدينا تركيبات فوق التعادل لكنها ضعيفة نسبياً\n"
        msg += f"💭 خياران: اعتمادهما بتردد أقل، أو إعادة تصميم الفلاتر\n"
    else:
        msg += f"❌ *لا توجد حافة حقيقية بأي تركيبة*\n"
        msg += f"🔄 نحتاج إعادة تصميم القواعد من الصفر (فلاتر جديدة)\n"

    msg += f"\n⏱️ انتهى في {time.time()-start:.0f} ثانية"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("TG غير معد - اطبع التقرير محلياً فقط")
        print(text)
        return
    # تقسيم إذا طويلاً جداً
    chunks = []
    lines = text.split("\n")
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)
    for i, chunk in enumerate(chunks):
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            payload = {"chat_id": TG_CHAT, "text": chunk, "parse_mode": "Markdown", "disable_web_page_preview": True}
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                log.info(f"✅ الجزء {i+1}/{len(chunks)} أُرسل")
            else:
                log.warning(f"TG: {r.status_code}")
            time.sleep(1)
        except Exception as e:
            log.error(f"TG error: {e}")

def main():
    report = build_matrix_report()
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    send_telegram(report)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.exception(f"fatal: {e}")
        raise
