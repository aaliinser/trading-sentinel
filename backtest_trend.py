#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest: Strategy 2 - Trend Following (EMA200 + EMA50/20 Pullback)
الاستراتيجية الخام كما وردت حرفياً — لم يُغيَّر أي شرط دخول.
نموذج الحسم: إغلاق الشمعة التالية (مطابق لمنطق بوتاتك الحالية).
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

# ═══════════════════════════════════════════════
# الإعدادات
# ═══════════════════════════════════════════════
SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURCHF=X",
    "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X",
    "EURAUD=X", "USDCHF=X", "AUDNZD=X", "GBPCAD=X"
]

INTERVAL = "1h"          # فريم الاستراتيجية
PERIOD = "730d"          # سنتان (الحد الأقصى المتاح لفريم الساعة لدى Yahoo)
WARMUP = 210             # تجاهل أول 210 شموع لضمان استقرار EMA200

PAYOUT = 0.85            # عائد المنصة
RISK_USD = 10.0          # مخاطرة الصفقة الواحدة
FLAT_THRESHOLD_PIPS = 0.1

OUT_JSON = "backtest_trend_result.json"
OUT_CSV = "backtest_trend_trades.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("BT")

try:
    TZ = ZoneInfo("Asia/Hebron")
except Exception:
    TZ = ZoneInfo("UTC")


# ═══════════════════════════════════════════════
# الاستراتيجية الخام — كما أرسلتها دون أي تعديل
# ═══════════════════════════════════════════════
def strategy_2_trend_following(df: pd.DataFrame):
    df = df.copy()
    df["ema200"] = df["close"].ewm(span=200, min_periods=200, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, min_periods=50, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, min_periods=20, adjust=False).mean()

    c = df.iloc[-1]
    if pd.isna(c["ema200"]) or pd.isna(c["ema50"]) or pd.isna(c["ema20"]):
        return "WAIT"

    close_p = float(c["close"])
    open_p = float(c["open"])
    high_p = float(c["high"])
    low_p = float(c["low"])
    ema200 = float(c["ema200"])
    ema50 = float(c["ema50"])
    ema20 = float(c["ema20"])

    if close_p > ema200 and ema20 > ema50 and low_p <= ema20 and close_p > open_p:
        return "CALL"
    elif close_p < ema200 and ema20 < ema50 and high_p >= ema20 and close_p < open_p:
        return "PUT"
    return "WAIT"


# ═══════════════════════════════════════════════
# محرك البيانات
# ═══════════════════════════════════════════════
def fetch(symbol: str):
    """جلب مع إعادة محاولة وتراجع تدريجي لمواجهة Rate Limit."""
    for attempt in range(5):
        try:
            df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL,
                                           auto_adjust=False, actions=False)
            if df is not None and not df.empty:
                break
        except Exception as e:
            log.warning(f"{symbol} attempt {attempt+1}: {type(e).__name__}")
            df = None
        time.sleep(5 * (attempt + 1))

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated()].sort_index()
    df.columns = [str(c).lower() for c in df.columns]
    need = ["open", "high", "low", "close"]
    if not all(k in df.columns for k in need):
        return None
    df = df[need].dropna()

    # حذف الشمعة الجارية غير المكتملة
    now = pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(minutes=60 if INTERVAL == "1h" else 15)
    while not df.empty and (df.index[-1] + step) > now:
        df = df.iloc[:-1]

    return df if len(df) > WARMUP + 50 else None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """نفس حساب المؤشرات مرة واحدة للأداء (متطابق رياضياً مع الدالة الخام)."""
    d = df.copy()
    d["ema200"] = d["close"].ewm(span=200, min_periods=200, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, min_periods=50, adjust=False).mean()
    d["ema20"] = d["close"].ewm(span=20, min_periods=20, adjust=False).mean()
    return d


def signal_at(row) -> str:
    """تطبيق شروط الدخول الخام على شمعة واحدة."""
    if pd.isna(row["ema200"]) or pd.isna(row["ema50"]) or pd.isna(row["ema20"]):
        return "WAIT"
    c, o, h, l = float(row["close"]), float(row["open"]), float(row["high"]), float(row["low"])
    e200, e50, e20 = float(row["ema200"]), float(row["ema50"]), float(row["ema20"])
    if c > e200 and e20 > e50 and l <= e20 and c > o:
        return "CALL"
    if c < e200 and e20 < e50 and h >= e20 and c < o:
        return "PUT"
    return "WAIT"


# ═══════════════════════════════════════════════
# محرك الباك تيست
# ═══════════════════════════════════════════════
def backtest_symbol(symbol: str, df: pd.DataFrame) -> list:
    d = add_indicators(df)
    pip_mult = 100.0 if "JPY" in symbol else 10000.0
    trades = []

    arr = d.reset_index()
    tcol = arr.columns[0]

    for i in range(WARMUP, len(arr) - 1):
        row = arr.iloc[i]
        sig = signal_at(row)
        if sig == "WAIT":
            continue

        nxt = arr.iloc[i + 1]
        entry_px = float(row["close"])
        exit_px = float(nxt["close"])
        diff = exit_px - entry_px
        pips = abs(diff) * pip_mult

        if pips < FLAT_THRESHOLD_PIPS:
            win = None
        else:
            win = (diff > 0) if sig == "CALL" else (diff < 0)

        ts = pd.Timestamp(row[tcol])
        local = ts.tz_convert(TZ)

        trades.append({
            "symbol": symbol,
            "signal_time_utc": ts.isoformat(),
            "exit_time_utc": pd.Timestamp(nxt[tcol]).isoformat(),
            "local_date": local.strftime("%Y-%m-%d"),
            "local_hour": int(local.hour),
            "weekday": int(local.weekday()),
            "type": sig,
            "entry_price": round(entry_px, 6),
            "exit_price": round(exit_px, 6),
            "win": win,
            "pips": round(pips, 2),
            "signed_pips": round((pips if win else (-pips if win is False else 0.0)), 2),
        })
    return trades


def stats(trades: list) -> dict:
    total = len(trades)
    wins = sum(1 for t in trades if t["win"] is True)
    losses = sum(1 for t in trades if t["win"] is False)
    voids = total - wins - losses
    valid = wins + losses
    wr = (wins / valid * 100.0) if valid else 0.0
    net = sum(t["signed_pips"] for t in trades)
    capital = wins * RISK_USD * PAYOUT - losses * RISK_USD
    return {"total": total, "wins": wins, "losses": losses, "voids": voids,
            "win_rate": round(wr, 2), "net_pips": round(net, 2),
            "capital_usd": round(capital, 2)}


def streaks(seq):
    mw = ml = cw = cl = 0
    for w in seq:
        if w is True:
            cw += 1; cl = 0
        elif w is False:
            cl += 1; cw = 0
        else:
            cw = cl = 0
        mw = max(mw, cw); ml = max(ml, cl)
    return mw, ml


def group_stats(trades, key):
    g = {}
    for t in trades:
        g.setdefault(t[key], []).append(t)
    return {str(k): stats(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def main():
    log.info("=" * 64)
    log.info("BACKTEST | Strategy 2 - Trend Following (EMA200/50/20 Pullback)")
    log.info(f"Interval={INTERVAL} | Period={PERIOD} | Payout={PAYOUT}")
    log.info("=" * 64)

    all_trades = []
    per_symbol = {}
    failed = []

    for sym in SYMBOLS:
        df = fetch(sym)
        if df is None:
            log.error(f"{sym}: NO DATA (skipped)")
            failed.append(sym)
            continue
        tr = backtest_symbol(sym, df)
        all_trades.extend(tr)
        per_symbol[sym] = stats(tr)
        s = per_symbol[sym]
        log.info(f"{sym:10s} candles={len(df):6d} trades={s['total']:4d} "
                 f"WR={s['win_rate']:5.1f}% net={s['net_pips']:+8.1f}")
        time.sleep(2)

    if not all_trades:
        log.critical("No trades generated — تحقق من البيانات أو الشروط.")
        return

    overall = stats(all_trades)
    mw, ml = streaks([t["win"] for t in all_trades])
    breakeven = 1.0 / (1.0 + PAYOUT) * 100.0

    by_pair = group_stats(all_trades, "symbol")
    by_hour = group_stats(all_trades, "local_hour")
    by_day = group_stats(all_trades, "weekday")
    by_type = group_stats(all_trades, "type")

    ranked_pairs = sorted(by_pair.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)
    ranked_hours = sorted(by_hour.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)

    report = {
        "strategy": "Strategy 2 - Trend Following (EMA200 + EMA50/20 Pullback)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "interval": INTERVAL,
        "period": PERIOD,
        "payout": PAYOUT,
        "breakeven_win_rate_pct": round(breakeven, 2),
        "symbols_ok": list(per_symbol.keys()),
        "symbols_failed": failed,
        "overall": overall,
        "max_win_streak": mw,
        "max_loss_streak": ml,
        "edge_vs_breakeven_pp": round(overall["win_rate"] - breakeven, 2),
        "by_pair": by_pair,
        "by_hour_gaza": by_hour,
        "by_weekday": by_day,
        "by_direction": by_type,
        "best_pairs": ranked_pairs[:5],
        "worst_pairs": ranked_pairs[-5:],
        "best_hours": ranked_hours[:5],
        "worst_hours": ranked_hours[-5:],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pd.DataFrame(all_trades).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print("النتيجة الإجمالية")
    print("=" * 64)
    print(f"إجمالي الصفقات   : {overall['total']}")
    print(f"رابحة / خاسرة    : {overall['wins']} / {overall['losses']}  (تعادل {overall['voids']})")
    print(f"نسبة الفوز       : {overall['win_rate']}%")
    print(f"نسبة التعادل     : {breakeven:.2f}%  (عند عائد {PAYOUT*100:.0f}%)")
    print(f"الحافة           : {report['edge_vs_breakeven_pp']:+.2f} نقطة مئوية")
    print(f"صافي النقاط      : {overall['net_pips']:+.1f} Pips")
    print(f"محاكاة رأس المال : {overall['capital_usd']:+.2f} USD  (مخاطرة {RISK_USD}$)")
    print(f"أطول ربح/خسارة   : {mw} / {ml}")
    print("-" * 64)
    print("أفضل 5 أزواج:")
    for k, v in ranked_pairs[:5]:
        print(f"  {k:10s} WR={v['win_rate']:5.1f}%  n={v['total']:4d}  net={v['net_pips']:+8.1f}")
    print("أسوأ 5 أزواج:")
    for k, v in ranked_pairs[-5:]:
        print(f"  {k:10s} WR={v['win_rate']:5.1f}%  n={v['total']:4d}  net={v['net_pips']:+8.1f}")
    print("-" * 64)
    print("أفضل 5 ساعات (غزة):")
    for k, v in ranked_hours[:5]:
        print(f"  {int(k):02d}:00  WR={v['win_rate']:5.1f}%  n={v['total']:4d}")
    print("=" * 64)
    print(f"\n✅ حُفظ: {OUT_JSON} و {OUT_CSV}")

    verdict = ("رابحة إحصائياً" if overall["win_rate"] > breakeven + 2
               else ("حدّية — لا تتفوق على التعادل بثقة" if overall["win_rate"] >= breakeven
                     else "خاسرة إحصائياً عند هذا العائد"))
    print(f"الحكم: {verdict}\n")


if __name__ == "__main__":
    main()
