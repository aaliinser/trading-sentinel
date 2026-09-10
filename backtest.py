#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
غيث — v29b: القياس الحقيقي (دخول عند الرفض فوراً)
=====================================================================
يقيس بالضبط ما يمكن تنفيذه حياً:
- الإشارة تُطلق عند إغلاق شمعة الرفض (بدون انتظار التأكيد)
- الدخول عند سعر إغلاق الرفض نفسه (السعر المتاح فعلياً)
- اختبار 3 فترات انتهاء: 10د / 15د / 20د
- مقارنة مع v28 الأصلي كمرجع
=====================================================================
"""
import os, sys, time, logging
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
RN_LARGE = 0.5
RN_SMALL = 0.005
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Backtest_v29b")

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
        a = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_P)
        if a is not None:
            df["ATR"] = a
    else:
        df["EMA_35"] = df["Close"].ewm(span=EMA_F, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=EMA_S, adjust=False).mean()
        pc = df["Close"].shift(1)
        tr = pd.concat([
            df["High"]-df["Low"],
            (df["High"]-pc).abs(),
            (df["Low"]-pc).abs()
        ], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1/ATR_P, min_periods=ATR_P).mean()
    df["RS"] = df["Low"].rolling(LVL_LB, min_periods=20).min()
    df["RR"] = df["High"].rolling(LVL_LB, min_periods=20).max()
    df["UWICK"] = df["High"] - df[["Open","Close"]].max(axis=1)
    df["LWICK"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    return df

# ============ قواعد v28 (للحظة المرشحة) ============
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

def check_rej(d5_view, level, dr):
    """يتحقق من شمعة الرفض فقط (بدون تأكيد)"""
    if len(d5_view) < 2:
        return False, None, None
    rej = d5_view.iloc[-1]
    prev = d5_view.iloc[-2]
    rej_close = float(rej["Close"])
    rej_body = abs(float(rej["Close"]) - float(rej["Open"]))
    rej_range = float(rej["High"]) - float(rej["Low"])
    if rej_range <= 0 or rej_body <= 0:
        return False, None, None
    br = rej_body / rej_range
    brej = br >= REJ_BODY
    t = TOUCH_TOL * float(rej["Close"])
    if dr == "CALL":
        touched = float(rej["Low"]) <= level + t
    else:
        touched = float(rej["High"]) >= level - t
    if not touched:
        return False, None, None
    if dr == "CALL":
        lw = float(rej.get("LWICK", 0)) if pd.notna(rej.get("LWICK")) else 0
        pin = lw >= WICK_BODY * rej_body
        eng = rej["Close"] > rej["Open"] and prev["Close"] < prev["Open"] and rej["Close"] >= prev["Open"] and rej["Open"] <= prev["Close"]
        rej_ok = (brej or pin or eng) and rej_close > level
    else:
        uw = float(rej.get("UWICK", 0)) if pd.notna(rej.get("UWICK")) else 0
        pin = uw >= WICK_BODY * rej_body
        eng = rej["Close"] < rej["Open"] and prev["Close"] > prev["Open"] and rej["Close"] <= prev["Open"] and rej["Open"] >= prev["Close"]
        rej_ok = (brej or pin or eng) and rej_close < level
    return rej_ok, rej_close, rej.name

def check_confirm(d5_view, level, dr, rej_close):
    """يتحقق من شمعة التأكيد (للمقارنة مع v28)"""
    if len(d5_view) < 3:
        return False, None
    conf = d5_view.iloc[-1]
    rej = d5_view.iloc[-2]
    prev = d5_view.iloc[-3]
    conf_close = float(conf["Close"])
    rej_body = abs(float(rej["Close"]) - float(rej["Open"]))
    rej_range = float(rej["High"]) - float(rej["Low"])
    if rej_range <= 0 or rej_body <= 0:
        return False, None
    br = rej_body / rej_range
    brej = br >= REJ_BODY
    t = TOUCH_TOL * float(rej["Close"])
    if dr == "CALL":
        touched = float(rej["Low"]) <= level + t
    else:
        touched = float(rej["High"]) >= level - t
    if not touched:
        return False, None
    if dr == "CALL":
        lw = float(rej.get("LWICK", 0)) if pd.notna(rej.get("LWICK")) else 0
        pin = lw >= WICK_BODY * rej_body
        eng = rej["Close"] > rej["Open"] and prev["Close"] < prev["Open"] and rej["Close"] >= prev["Open"] and rej["Open"] <= prev["Close"]
        rej_ok = (brej or pin or eng) and float(rej["Close"]) > level
        conf_ok = conf_close > float(rej["Close"])
    else:
        uw = float(rej.get("UWICK", 0)) if pd.notna(rej.get("UWICK")) else 0
        pin = uw >= WICK_BODY * rej_body
        eng = rej["Close"] < rej["Open"] and prev["Close"] > prev["Open"] and rej["Close"] <= prev["Open"] and rej["Open"] >= prev["Close"]
        rej_ok = (brej or pin or eng) and float(rej["Close"]) < level
        conf_ok = conf_close < float(rej["Close"])
    return rej_ok and conf_ok, conf_close

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

# ============ جمع اللحظات المرشحة ============
def collect_candidates(sym):
    """لحظة = شمعة 5m استوفت v28 بدون شرط التأكيد"""
    log.info(f"=== {sym} ===")
    d15 = fetch(sym, SCAN_TF, f"{HISTORY_DAYS}d")
    d5  = fetch(sym, SNIPER_TF, "60d")
    d1h = fetch(sym, TREND_TF, f"{HISTORY_DAYS}d")
    if d15 is None or d5 is None or d1h is None:
        return []
    d15 = add_indicators(d15)
    d5  = add_indicators(d5)
    d1h = add_indicators(d1h)
    if d15 is None or d5 is None or d1h is None:
        return []
    if len(d15) < 300 or len(d5) < 1000 or len(d1h) < 100:
        return []

    candidates = []
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
        rej_ok, rej_close, rej_ts = check_rej(d5_view, level, dr)
        if not rej_ok:
            continue
        if not impulse_ok(d5_view, dr):
            continue
        if not deviation_ok(level, rej_close, dr):
            continue
        # تحقق أيضاً من التأكيد (للإحصاء المقارن مع v28)
        d5_view_conf = d5.iloc[max(0, i-20):i+2] if i+1 < len(d5) else None
        conf_ok = False
        conf_close = None
        if d5_view_conf is not None and len(d5_view_conf) >= 3:
            conf_ok, conf_close = check_confirm(d5_view_conf, level, dr, rej_close)
        # أسعار الخروج: 10/15/20 دقيقة بعد شمعة الرفض
        end_10 = i + 2   # شمعتان 5m بعد الرفض = 10د
        end_15 = i + 3   # 3 شموع = 15د
        end_20 = i + 4   # 4 شموع = 20د
        exit_10 = float(d5.iloc[end_10]["Close"]) if end_10 < len(d5) else None
        exit_15 = float(d5.iloc[end_15]["Close"]) if end_15 < len(d5) else None
        exit_20 = float(d5.iloc[end_20]["Close"]) if end_20 < len(d5) else None
        candidates.append({
            "time": cur_time,
            "dr": dr,
            "level": level,
            "rej_close": rej_close,
            "conf_close": conf_close,
            "conf_ok": conf_ok,
            "exit_10": exit_10,
            "exit_15": exit_15,
            "exit_20": exit_20,
        })
    log.info(f"{sym}: {len(candidates)} لحظة مرشحة")
    return candidates

# ============ تقييم تركيبة ============
def evaluate(candidates, entry_mode, expiry):
    """
    entry_mode: "reject" | "confirm"
    expiry: 10 | 15 | 20
    """
    trades = []
    for c in candidates:
        if entry_mode == "reject":
            entry = c["rej_close"]
        else:
            if not c["conf_ok"] or c["conf_close"] is None:
                continue
            entry = c["conf_close"]
        exit_key = f"exit_{expiry}"
        if c[exit_key] is None:
            continue
        dr = c["dr"]
        win = (c[exit_key] > entry) if dr == "CALL" else (c[exit_key] < entry)
        trades.append({
            "time": c["time"],
            "dr": dr,
            "win": win,
            "entry": entry,
            "exit": c[exit_key],
        })
    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = round(100 * wins / total, 2) if total > 0 else 0
    pnl = round(wins * STAKE * PAYOUT - (total - wins) * STAKE, 2)
    return {"total": total, "wins": wins, "losses": total - wins, "wr": wr, "pnl": pnl, "trades": trades}

# ============ اختبار الصلابة ============
def robustness_check(trades):
    if len(trades) < 20:
        return False, 0, 0
    mid_time = trades[len(trades)//2]["time"]
    first = [t for t in trades if t["time"] < mid_time]
    second = [t for t in trades if t["time"] >= mid_time]
    if len(first) < 10 or len(second) < 10:
        return False, 0, 0
    wr1 = 100 * sum(1 for t in first if t["win"]) / len(first)
    wr2 = 100 * sum(1 for t in second if t["win"]) / len(second)
    return wr1 >= 52.6 and wr2 >= 52.6, round(wr1, 1), round(wr2, 1)

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

def fmt_sym(s):
    b = s.replace("=X", "")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

# ============ بناء التقرير ============
def build_report():
    log.info(f"🚀 بدء v29b: {len(SYMBOLS)} زوجاً × {HISTORY_DAYS} يوم")
    start = time.time()

    all_candidates = {}
    for sym in SYMBOLS:
        try:
            all_candidates[sym] = collect_candidates(sym)
            time.sleep(1)
        except Exception as e:
            log.error(f"{sym}: {e}")
            all_candidates[sym] = []

    # 4 تركيبات:
    # A) v28 الأصلي (للمقارنة): دخول عند التأكيد + 15د
    # B) v29b-10: دخول عند الرفض + 10د
    # C) v29b-15: دخول عند الرفض + 15د  ← المرجح الأفضل
    # D) v29b-20: دخول عند الرفض + 20د
    combos = [
        ("confirm", 15, "v28 الأصلي (مرجع)",   "دخول عند التأكيد | 15د"),
        ("reject",  10, "v29b | 10 دقائق",      "دخول عند الرفض | 10د"),
        ("reject",  15, "v29b | 15 دقيقة",      "دخول عند الرفض | 15د"),
        ("reject",  20, "v29b | 20 دقيقة",      "دخول عند الرفض | 20د"),
    ]

    results = []
    for entry_mode, expiry, label, short in combos:
        total_t = 0
        total_w = 0
        total_pnl = 0.0
        by_sym = {}
        all_trades = []
        for sym, cands in all_candidates.items():
            if not cands:
                continue
            r = evaluate(cands, entry_mode, expiry)
            total_t += r["total"]
            total_w += r["wins"]
            total_pnl += r["pnl"]
            by_sym[sym] = r
            all_trades.extend(r["trades"])
        wr = round(100 * total_w / total_t, 2) if total_t > 0 else 0
        robust, wr1, wr2 = robustness_check(all_trades)
        sessions = session_analysis(all_trades) if all_trades else {}
        ranked = sorted(by_sym.items(), key=lambda x: x[1]["wr"] if x[1]["total"] >= 10 else -1, reverse=True)
        top3 = [(s, d) for s, d in ranked[:3] if d["total"] >= 10]
        signals_per_day = round(total_t / HISTORY_DAYS, 1)
        best_session = max(sessions.items(), key=lambda x: (100*x[1]["w"]/x[1]["t"] if x[1]["t"] else 0)) if sessions else None
        results.append({
            "label": label,
            "short": short,
            "entry_mode": entry_mode,
            "expiry": expiry,
            "total": total_t,
            "wins": total_w,
            "wr": wr,
            "pnl": round(total_pnl, 2),
            "robust": robust,
            "wr_h1": wr1,
            "wr_h2": wr2,
            "top3": top3,
            "best_session": best_session,
            "sessions": sessions,
            "signals_per_day": signals_per_day,
        })

    # بناء الرسالة
    msg = f"🔬 *v29b — القياس الحقيقي*\n"
    msg += f"({len(SYMBOLS)} زوجاً × {HISTORY_DAYS} يوم)\n\n"

    msg += f"📋 *المقارنة الحاسمة:*\n"
    msg += f"```\n"
    msg += f"{'التركيبة':<26} {'إشارات/يوم':>10} {'الفوز':>7} {'الربح':>10} {'صلب؟':>5}\n"
    msg += f"{'-'*26} {'-'*10} {'-'*7} {'-'*10} {'-'*5}\n"
    for r in results:
        robust_mark = "✅" if r["robust"] else "❌"
        pnl_mark = f"{r['pnl']:+.0f}$"
        msg += f"{r['short']:<26} {r['signals_per_day']:>10.1f} {r['wr']:>6.1f}% {pnl_mark:>10} {robust_mark:>5}\n"
    msg += f"```\n\n"

    # التركيز على التركيبات الثلاث الجديدة
    new_combos = [r for r in results if r["entry_mode"] == "reject"]
    winners = [r for r in new_combos if r["robust"] and r["wr"] >= 55]
    marginal = [r for r in new_combos if r["robust"] and 52.6 <= r["wr"] < 55]
    v28_ref = next(r for r in results if r["entry_mode"] == "confirm")

    msg += f"📊 *مقارنة v29b بـ v28 الأصلي:*\n"
    for r in new_combos:
        delta_wr = round(r["wr"] - v28_ref["wr"], 1)
        delta_pnl = round(r["pnl"] - v28_ref["pnl"], 0)
        delta_sign = "+" if delta_wr > 0 else ""
        delta_pnl_sign = "+" if delta_pnl > 0 else ""
        msg += f"• {r['short']}: {delta_sign}{delta_wr} نقطة | {delta_pnl_sign}{delta_pnl:.0f}$ ربح إضافي\n"

    msg += f"\n"
    if winners:
        best = max(winners, key=lambda x: (x["wr"], x["pnl"]))
        msg += f"🏆 *التركيبة الفائزة: {best['short']}*\n\n"
        msg += f"🎯 *الأرقام الحقيقية القابلة للتنفيذ:*\n"
        msg += f"• نسبة الفوز: *{best['wr']}%*\n"
        msg += f"• الربح 60 يوم: *{best['pnl']:+.2f}$*\n"
        msg += f"• الإشارات/يوم: *{best['signals_per_day']}* (كل الأزواج)\n"
        msg += f"• الصلابة: النصف الأول {best['wr_h1']}% | الثاني {best['wr_h2']}%\n\n"

        if best["top3"]:
            msg += f"🥇 *أفضل 3 أزواج:*\n"
            for s, d in best["top3"]:
                msg += f"  • *{fmt_sym(s)}*: {d['wr']}% ({d['total']} صفقة)\n"

        if best["best_session"] and best["best_session"][1]["t"] >= 20:
            s_name = best["best_session"][0]
            s_wr = round(100*best["best_session"][1]["w"]/best["best_session"][1]["t"], 1)
            s_t = best["best_session"][1]["t"]
            msg += f"\n⏰ *أفضل جلسة:* {s_name} ({s_wr}% — {s_t} صفقة)\n"

        # الإشارات/يوم للأزواج القوية فقط (فلترة)
        top_symbols = [s for s, d in best["top3"][:3]]
        if top_symbols:
            top_trades = sum(by_sym[s]["total"] for s in top_symbols for r2 in [best] for by_sym in [{s2: evaluate(all_candidates[s2], best["entry_mode"], best["expiry"]) for s2 in SYMBOLS if s2 in all_candidates}])
        # نحسب بدقة
        filtered_signals = 0
        filtered_wins = 0
        for s in top_symbols:
            if s in all_candidates:
                r2 = evaluate(all_candidates[s], best["entry_mode"], best["expiry"])
                filtered_signals += r2["total"]
                filtered_wins += r2["wins"]
        if filtered_signals > 0:
            filtered_wr = round(100 * filtered_wins / filtered_signals, 1)
            filtered_per_day = round(filtered_signals / HISTORY_DAYS, 1)
            msg += f"\n🎯 *بعد الفلترة (أفضل 3 أزواج فقط):*\n"
            msg += f"• نسبة الفوز: *{filtered_wr}%*\n"
            msg += f"• الإشارات/يوم: *{filtered_per_day}* (واقعية أكثر)\n"

        msg += f"\n✅ *توصية:* نعتمد هذه التركيبة كـ v29\n"

    elif marginal:
        msg += f"⚠️ *تركيبات هامشية:*\n"
        for r in marginal:
            msg += f"• {r['short']}: {r['wr']}% ({r['pnl']:+.0f}$)\n"
        msg += f"\n💭 قرار: نعتمدها بحذر، أو ننتظر أسبوع بيانات إضافية\n"

    else:
        msg += f"❌ *لا تركيبة رابحة صلبة*\n"
        msg += f"القياس الحقيقي أظهر أن دخول الرفض فوراً لا يعطي الحافة المتوقعة\n"
        msg += f"💭 نحتاج إعادة تصميم\n"

    msg += f"\n⏱️ انتهى في {time.time()-start:.0f} ثانية"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("TG غير معد")
        print(text)
        return
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
                log.info(f"✅ الجزء {i+1}/{len(chunks)}")
            time.sleep(1)
        except Exception as e:
            log.error(f"TG: {e}")

def main():
    report = build_report()
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
