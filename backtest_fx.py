#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — روح BLW: عزل مكوّنات RSI20 + ستوكاستك (5 نسخ)
فريم 30د + انتهاء 15د
"""
import os, sys, time, logging
import numpy as np, pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
REF_H2 = 56.1

VARIANTS = [
    (1, "RSI20 only"),
    (2, "Stoch cross only"),
    (3, "RSI OR Stoch"),
    (4, "RSI+Stoch zone"),
    (5, "Looser 25/75"),
]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("BLW_Soul")

def fetch(sym, iv, period):
    for attempt in range(1, 5):
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=30)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
            if len(needed) < 4:
                raise ValueError("missing columns")
            df = df[needed]
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(inplace=True)
            df = df[(df["High"] >= df["Low"]) & (df["Open"] > 0) & (df["Close"] > 0)]
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(3 * attempt)
    return None

def rsi_ser(c, p):
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    ratio = ag / al.replace(0, np.nan)
    rsi = 100 - (100 / (1 + ratio))
    return rsi.fillna(50)

def stochastic(df, k=14, smooth=3, d=3):
    hs = df["High"]
    ls = df["Low"]
    cs = df["Close"]
    ll = ls.rolling(k, min_periods=k).min()
    hh = hs.rolling(k, min_periods=k).max()
    denom = (hh - ll).replace(0, np.nan)
    raw_k = 100 * (cs - ll) / denom
    k_line = raw_k.rolling(smooth, min_periods=1).mean()
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line, d_line

def prepare(sym):
    d15 = fetch(sym, "15m", f"{HISTORY_DAYS}d")
    d30 = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    if d15 is None or d30 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d15 = d15[d15.index + pd.Timedelta(minutes=15) <= now]
    d30 = d30[d30.index + pd.Timedelta(minutes=30) <= now]
    if len(d30) < 200 or len(d15) < 100:
        return None
    cs = d30["Close"]
    rsi20 = rsi_ser(cs, 20).to_numpy(dtype=float)
    k_line, d_line = stochastic(d30, 14, 3, 3)
    return {
        "times": d30.index,
        "c": cs.to_numpy(dtype=float),
        "rsi20": rsi20,
        "stk_k": k_line.to_numpy(dtype=float),
        "stk_d": d_line.to_numpy(dtype=float),
        "c15map": d15["Close"].to_dict(),
    }

def evaluate(data, vid):
    t = data["times"]
    c = data["c"]
    rsi = data["rsi20"]
    k = data["stk_k"]
    d = data["stk_d"]
    c15map = data["c15map"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if np.isnan(rsi[i]) or np.isnan(k[i]) or np.isnan(d[i]):
            continue
        if np.isnan(k[i-1]) or np.isnan(d[i-1]):
            continue
        r_lo = rsi[i] <= 20.0
        r_hi = rsi[i] >= 80.0
        r_lo2 = rsi[i] <= 25.0
        r_hi2 = rsi[i] >= 75.0
        kz_lo = k[i] < 20.0
        kz_hi = k[i] > 80.0
        kz_lo2 = k[i] < 25.0
        kz_hi2 = k[i] > 75.0
        x_up = (k[i-1] < d[i-1]) and (k[i] > d[i])
        x_dn = (k[i-1] > d[i-1]) and (k[i] < d[i])
        call = False
        put = False
        if vid == 1:
            call = r_lo
            put = r_hi
        elif vid == 2:
            call = x_up and kz_lo
            put = x_dn and kz_hi
        elif vid == 3:
            call = r_lo or (x_up and kz_lo)
            put = r_hi or (x_dn and kz_hi)
        elif vid == 4:
            call = r_lo and kz_lo
            put = r_hi and kz_hi
        elif vid == 5:
            call = r_lo2 and x_up and kz_lo2
            put = r_hi2 and x_dn and kz_hi2
        if not (call or put):
            continue
        T = t[i] + pd.Timedelta(minutes=15)
        exit_px = c15map.get(T)
        if exit_px is None:
            continue
        if isinstance(exit_px, float) and np.isnan(exit_px):
            continue
        entry = c[i]
        if call:
            win = exit_px > entry
        else:
            win = exit_px < entry
        trades.append((t[i], bool(win)))
    return trades

def stats(trades):
    if not trades:
        return None
    total = len(trades)
    wins = sum(1 for _, w in trades if w)
    wr = round(100 * wins / total, 2)
    pnl = round(wins * STAKE * PAYOUT - (total - wins) * STAKE, 2)
    return {"total": total, "wins": wins, "wr": wr, "pnl": pnl}

def robustness(trades):
    if len(trades) < 200:
        return False, 0.0, 0.0
    trades = sorted(trades, key=lambda x: x[0])
    mid = len(trades) // 2
    s1 = stats(trades[:mid])
    s2 = stats(trades[mid:])
    if not s1 or not s2:
        return False, 0.0, 0.0
    ok = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return ok, s1["wr"], s2["wr"]

def build_report():
    log.info("بدء اختبار روح BLW (5 نسخ)")
    start = time.time()
    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            log.info(f"{sym}: جاهز")
        except Exception as e:
            log.error(f"{sym}: {e}")
            data_by_sym[sym] = None
        time.sleep(1)
    rows = []
    for vid, label in VARIANTS:
        trades = []
        for sym, data in data_by_sym.items():
            if data is None:
                continue
            trades.extend(evaluate(data, vid))
        st = stats(trades)
        if not st:
            rows.append((vid, label, 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, label, st["total"], st["wr"], rob, w1, w2, per_day))
        log.info(f"نسخة {vid}: {st['total']} صفقة {st['wr']}%")
    msg = f"🧬 *روح BLW — عزل المكوّنات*\n(5 أزواج × 60 يوم × 30د × 15د)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'النسخة':<18}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>4}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<3}{r[1]:<18}{r[2]:>7}{r[3]:>7.1f}%{r[7]:>6}{mark:>4}\n"
    msg += "```\n\n"
    msg += f"🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[2] >= 200:
            msg += f"• نسخة {r[0]}: {r[5]:.1f}% | {r[6]:.1f}%\n"
    valid = [r for r in rows if r[2] >= 300]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += f"• أفضل مكوّن: *{best[3]}%* ({best[1]})\n"
        cands = [r for r in valid if r[4] and r[3] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *مكوّن حقيقي متفوق:*\n"
            for r in cands:
                msg += f"• نسخة {r[0]}: *{r[3]}%*\n"
            msg += f"\n⏳ تستحق ديمو خاصة بعد ديمو H2\n"
        else:
            msg += f"\n✅ *لا مكوّن يتفوق على H2* — الروح نفسها ميتة\n"
    else:
        msg += f"\n⚠️ لا نسخة بلغت 300 صفقة — الفكرة نادرة حتى بعد التخفيف\n"
    msg += f"\n🔒 H2 والديمو لا تتأثران\n"
    msg += f"\n⏱️ {time.time()-start:.0f}ث"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            log.info("✅ أُرسل")
        else:
            log.warning(f"TG {r.status_code}")
    except Exception as e:
        log.error(f"TG: {e}")

def main():
    report = build_report()
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    send_telegram(report)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.exception(f"fatal: {e}")
        raise    c = data["c"]
    rsi = data["rsi20"]
    k = data["stk_k"]
    d = data["stk_d"]
    c15map = data["c15map"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if np.isnan(rsi[i]) or np.isnan(k[i]) or np.isnan(d[i]):
            continue
        if np.isnan(k[i-1]) or np.isnan(d[i-1]):
            continue
        T = t[i] + pd.Timedelta(minutes=15)
        exit_px = c15map.get(T)
        if exit_px is None:
            continue
        if isinstance(exit_px, float) and np.isnan(exit_px):
            continue
        entry = c[i]
        dr = None
        if rsi[i] <= 20 and k[i-1] < d[i-1] and k[i] > d[i] and k[i] < 20:
            dr = "CALL"
        elif rsi[i] >= 80 and k[i-1] > d[i-1] and k[i] < d[i] and k[i] > 80:
            dr = "PUT"
        if dr is None:
            continue
        if dr == "CALL":
            win = exit_px > entry
        else:
            win = exit_px < entry
        trades.append((t[i], bool(win)))
    return trades

def stats(trades):
    if not trades:
        return None
    total = len(trades)
    wins = sum(1 for _, w in trades if w)
    wr = round(100 * wins / total, 2)
    pnl = round(wins * STAKE * PAYOUT - (total - wins) * STAKE, 2)
    return {"total": total, "wins": wins, "wr": wr, "pnl": pnl}

def robustness(trades):
    if len(trades) < 200:
        return False, 0.0, 0.0
    trades = sorted(trades, key=lambda x: x[0])
    mid = len(trades) // 2
    s1 = stats(trades[:mid])
    s2 = stats(trades[mid:])
    if not s1 or not s2:
        return False, 0.0, 0.0
    ok = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return ok, s1["wr"], s2["wr"]

def build_report():
    log.info("بدء اختبار BLW (محصنة v2)")
    start = time.time()
    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            cnt = 0
            if data_by_sym[sym] is not None:
                cnt = len(data_by_sym[sym]["c"])
            log.info(f"{sym}: جاهز ({cnt} شمعة)")
        except Exception as e:
            log.error(f"{sym}: {e}")
            data_by_sym[sym] = None
        time.sleep(1)
    trades = []
    for sym, data in data_by_sym.items():
        if data is None:
            continue
        try:
            trades.extend(evaluate(data))
        except Exception as e:
            log.error(f"{sym} evaluate: {e}")
    st = stats(trades)
    msg = f"🎥 *اختبار BLW (محصنة v2)*\n(5 أزواج × 60 يوم × 30د × 15د)\n\n"
    if not st:
        msg += "❌ *لا صفقات*\n"
        msg += "القواعد: RSI(20) 80/20 + تقاطع ستوكاستك عند نفس المستوى\n"
        msg += "النتيجة: الشروط نادرة جداً على فريم 30د\n"
    else:
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        msg += "```\n"
        msg += f"صفقات: {st['total']}\n"
        msg += f"فوز: {st['wr']}%\n"
        msg += f"لكل يوم: {per_day}\n"
        msg += f"صلبة: {'Y' if rob else 'N'}\n"
        msg += "```\n\n"
        msg += f"📋 القواعد المختبرة:\n"
        msg += f"• RSI فترة 20 بمستويات 80/20\n"
        msg += f"• ستوكاستك 14/3/3\n"
        msg += f"• CALL: RSI تحت 20 + تقاطع ستوكاستك صعوداً\n"
        msg += f"• PUT: RSI فوق 80 + تقاطع ستوكاستك هبوطاً\n"
        if st["total"] >= 200:
            msg += f"\n🧪 صلابة (نصف|نصف): {w1:.1f}% | {w2:.1f}%\n"
        msg += f"\n📏 *المقارنة:*\n"
        msg += f"• مرجع H2: *{REF_H2}%*\n"
        msg += f"• نتيجة BLW: *{st['wr']}%* على {st['total']} صفقة\n"
        if st["total"] < 300:
            msg += f"\n⚠️ عينة قليلة — النتيجة استرشادية فقط\n"
        elif rob and st["wr"] >= REF_H2 + 2.0:
            msg += f"\n🏆 *BLW تتفوق على H2* — تستحق ديمو خاصة\n"
        elif rob and st["wr"] >= BREAKEVEN:
            msg += f"\n🟡 رابحة لكن أضعف من H2 — لا تستبدل\n"
        else:
            msg += f"\n🔴 *فاشلة* — تحت التعادل = خسارة مؤكدة\n"
    msg += f"\n🔒 H2 والديمو لا تتأثران\n"
    msg += f"\n⏱️ {time.time()-start:.0f}ث"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            log.info("✅ أُرسل")
        else:
            log.warning(f"TG {r.status_code}")
    except Exception as e:
        log.error(f"TG: {e}")

def main():
    report = build_report()
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    send_telegram(report)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.exception(f"fatal: {e}")
        raise
