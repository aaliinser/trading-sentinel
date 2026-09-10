#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
غيث — باك تست v28 الشامل (20 زوجاً × 60 يوم)
=====================================================================
يعيد تشغيل قواعد v28 الرشيقة على التاريخ الكامل لكل زوج
ويخرج تقريراً مفصلاً لكل زوج + ملخصاً عاماً + توصيات
=====================================================================
"""
import os, sys, time, json, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
log = logging.getLogger("Backtest")

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

# ============ قواعد v28 ============
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
    """يفحص آخر 3 شموع 5m: رفض + تأكيد"""
    if len(d5_view) < 3:
        return False, None
    conf, rej, prev = d5_view.iloc[-1], d5_view.iloc[-2], d5_view.iloc[-3]
    rej_close = float(rej["Close"])
    rej_body = abs(float(rej["Close"]) - float(rej["Open"]))
    rej_range = float(rej["High"]) - float(rej["Low"])
    if rej_range <= 0 or rej_body <= 0:
        return False, None
    br = rej_body / rej_range
    brej = br >= REJ_BODY
    # اللمس
    t = TOUCH_TOL * float(rej["Close"])
    if dr == "CALL":
        touched = float(rej["Low"]) <= level + t
    else:
        touched = float(rej["High"]) >= level - t
    if not touched:
        return False, None
    # الرفض
    if dr == "CALL":
        lw = float(rej.get("LWICK", 0)) if pd.notna(rej.get("LWICK")) else 0
        pin = lw >= WICK_BODY * rej_body
        eng = rej["Close"] > rej["Open"] and prev["Close"] < prev["Open"] and rej["Close"] >= prev["Open"] and rej["Open"] <= prev["Close"]
        rej_ok = (brej or pin or eng) and rej_close > level
        conf_ok = float(conf["Close"]) > rej_close
    else:
        uw = float(rej.get("UWICK", 0)) if pd.notna(rej.get("UWICK")) else 0
        pin = uw >= WICK_BODY * rej_body
        eng = rej["Close"] < rej["Open"] and prev["Close"] > prev["Open"] and rej["Close"] <= prev["Open"] and rej["Open"] >= prev["Close"]
        rej_ok = (brej or pin or eng) and rej_close < level
        conf_ok = float(conf["Close"]) < rej_close
    return rej_ok and conf_ok, conf.name

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

# ============ المحاكاة ============
def run_backtest(sym):
    log.info(f"=== {sym} ===")
    d15 = fetch(sym, SCAN_TF, "7d" if HISTORY_DAYS <= 7 else f"{HISTORY_DAYS}d")
    d5  = fetch(sym, SNIPER_TF, "60d")
    d1h = fetch(sym, TREND_TF, f"{HISTORY_DAYS}d")
    if d15 is None or d5 is None or d1h is None:
        log.warning(f"{sym}: فشل جلب البيانات")
        return None
    d15 = add_indicators(d15)
    d5  = add_indicators(d5)
    d1h = add_indicators(d1h)
    if d15 is None or d5 is None or d1h is None:
        return None
    if len(d15) < 300 or len(d5) < 1000 or len(d1h) < 100:
        log.warning(f"{sym}: بيانات غير كافية")
        return None

    trades = []
    # نمشي على كل شمعة 5m مغلقة
    for i in range(10, len(d5)):
        cur_time = d5.index[i]
        # أوجد أقرب شمعة 15m و 1h <= cur_time
        m15 = d15[d15.index <= cur_time]
        m1h = d1h[d1h.index <= cur_time]
        if len(m15) < 5 or len(m1h) < 3:
            continue
        last_15 = m15.iloc[-1]
        prev_15 = m15.iloc[-2]
        last_h1 = m1h.iloc[-1]
        # 1) ترند
        dr = trend_aligned(last_15, prev_15, last_h1)
        if dr is None:
            continue
        # 2) EMA
        if not ema_distance_ok(last_15):
            continue
        # 3) مستوى
        level = find_level(last_15, dr)
        if level is None:
            continue
        # 4) نافذة 5m
        d5_view = d5.iloc[max(0, i-20):i+1]
        if len(d5_view) < 10:
            continue
        # 5) لمس + رفض + تأكيد
        ok, sig_time = touch_rej_confirm(d5_view, level, dr)
        if not ok or sig_time is None:
            continue
        # 6) اندفاع
        if not impulse_ok(d5_view, dr):
            continue
        # 7) انحراف
        entry_price = float(d5_view.iloc[-1]["Close"])
        if not deviation_ok(level, entry_price, dr):
            continue
        # 8) محاكاة انتهاء 15 دقيقة = 3 شموع 5m بعد
        end_idx = i + 3
        if end_idx >= len(d5):
            continue
        exit_price = float(d5.iloc[end_idx]["Close"])
        win = (exit_price > entry_price) if dr == "CALL" else (exit_price < entry_price)
        trades.append({
            "time": cur_time,
            "dr": dr,
            "level": level,
            "entry": entry_price,
            "exit": exit_price,
            "win": win
        })
    return trades

# ============ التحليل ============
def analyze(sym, trades):
    if not trades:
        return {"symbol": sym, "trades": 0, "wins": 0, "wr": 0, "pnl": 0,
                "call_w": 0, "call_t": 0, "put_w": 0, "put_t": 0,
                "by_session": {}}
    wins = sum(1 for t in trades if t["win"])
    call_trades = [t for t in trades if t["dr"] == "CALL"]
    put_trades = [t for t in trades if t["dr"] == "PUT"]
    pnl = wins * STAKE * PAYOUT - (len(trades) - wins) * STAKE
    by_session = {}
    for t in trades:
        h = t["time"].hour
        if 2 <= h < 6:
            s = "آسيا"
        elif 7 <= h < 11:
            s = "لندن-افتتاح"
        elif 11 <= h < 15:
            s = "لندن"
        elif 15 <= h < 20:
            s = "نيويورك"
        else:
            s = "هادئة"
        if s not in by_session:
            by_session[s] = {"w": 0, "t": 0}
        by_session[s]["t"] += 1
        if t["win"]:
            by_session[s]["w"] += 1
    return {
        "symbol": sym,
        "trades": len(trades),
        "wins": wins,
        "wr": round(100 * wins / len(trades), 1),
        "pnl": round(pnl, 2),
        "call_w": sum(1 for t in call_trades if t["win"]),
        "call_t": len(call_trades),
        "put_w": sum(1 for t in put_trades if t["win"]),
        "put_t": len(put_trades),
        "by_session": by_session
    }

# ============ التقرير ============
def fmt_sym(s):
    b = s.replace("=X", "")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def build_report(results):
    valid = [r for r in results if r["trades"] > 0]
    if not valid:
        return "❌ لا توجد بيانات كافية لأي زوج"
    total_t = sum(r["trades"] for r in valid)
    total_w = sum(r["wins"] for r in valid)
    total_pnl = sum(r["pnl"] for r in valid)
    overall_wr = round(100 * total_w / total_t, 1) if total_t > 0 else 0

    # ترتيب حسب نسبة الفوز (مع حد أدنى 10 صفقات)
    ranked = sorted([r for r in valid if r["trades"] >= 10], key=lambda x: x["wr"], reverse=True)
    # ترتيب حسب عدد الصفقات
    most_active = sorted(valid, key=lambda x: x["trades"], reverse=True)[:5]

    # تجميع الجلسات
    sessions = {}
    for r in valid:
        for s, d in r["by_session"].items():
            if s not in sessions:
                sessions[s] = {"w": 0, "t": 0}
            sessions[s]["t"] += d["t"]
            sessions[s]["w"] += d["w"]
    session_lines = []
    for s, d in sorted(sessions.items(), key=lambda x: (100*x[1]["w"]/x[1]["t"] if x[1]["t"] else 0), reverse=True):
        if d["t"] >= 5:
            wr = round(100 * d["w"] / d["t"], 1)
            medal = "🏆" if wr >= 58 else ("✅" if wr >= 53 else "⚠️")
            session_lines.append(f"• {medal} {s}: {wr}% ({d['t']} صفقة)")

    verdict = "🟢 استراتيجية رابحة" if overall_wr >= 55 else ("🟡 هامشية" if overall_wr >= 52.6 else "🔴 ضعيفة")

    msg = f"📊 *تقرير باك تست v28* (60 يوم)\n\n"
    msg += f"🎯 *النتيجة العامة:*\n"
    msg += f"• صفقات: {total_t}\n"
    msg += f"• فوز: {total_w} | خسارة: {total_t - total_w}\n"
    msg += f"• *نسبة الفوز: {overall_wr}%*\n"
    msg += f"• صافي الربح: {total_pnl:+.2f}$\n"
    msg += f"• الحكم: {verdict}\n\n"

    msg += f"📈 *ترتيب الأزواج (الأقوى):*\n"
    for i, r in enumerate(ranked[:8], 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}."))
        msg += f"{medal} *{fmt_sym(r['symbol'])}*: {r['wr']}% ({r['trades']} صفقة) — {r['pnl']:+.0f}$\n"

    msg += f"\n🔻 *الأزواج الأضعف:*\n"
    for r in ranked[-3:] if len(ranked) > 5 else []:
        msg += f"• ❌ {fmt_sym(r['symbol'])}: {r['wr']}%\n"

    msg += f"\n⏰ *حسب الجلسات:*\n"
    msg += "\n".join(session_lines) if session_lines else "• لا بيانات كافية"

    msg += f"\n💡 *التوصيات:*\n"
    if ranked and ranked[0]["wr"] >= 55:
        msg += f"• ركّز على: {', '.join(fmt_sym(r['symbol']) for r in ranked[:3])}\n"
    if session_lines:
        best_s = max(sessions.items(), key=lambda x: (100*x[1]["w"]/x[1]["t"] if x[1]["t"] else 0))
        msg += f"• أفضل جلسة: *{best_s[0]}*\n"
    if overall_wr >= 55:
        msg += f"• ✅ ابدأ التداول الحقيقي بثقة\n"
    elif overall_wr >= 52.6:
        msg += f"• ⚠️ هامشية — نحتاج تعديل بوابة\n"
    else:
        msg += f"• ❌ نحتاج إعادة تصميم القواعد\n"

    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("TG غير معد - اطبع التقرير محلياً فقط")
        print(text)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            log.info("✅ التقرير أُرسل على تليجرام")
        else:
            log.warning(f"TG: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        log.error(f"TG error: {e}")

# ============ الرئيسي ============
def main():
    log.info(f"🚀 بدء باك تست v28 على {len(SYMBOLS)} زوجاً × {HISTORY_DAYS} يوم")
    start = time.time()
    results = []
    for sym in SYMBOLS:
        try:
            trades = run_backtest(sym)
            if trades is not None:
                r = analyze(sym, trades)
                results.append(r)
                log.info(f"{fmt_sym(sym)}: {r['trades']} صفقة، {r['wr']}%")
            time.sleep(1)
        except Exception as e:
            log.error(f"{sym}: {e}")
    report = build_report(results)
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    send_telegram(report)
    log.info(f"انتهى في {time.time()-start:.0f} ثانية")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.exception(f"fatal: {e}")
        raise
