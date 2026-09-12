#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — العشرة الكبار على الفريمات الصغيرة
فريمات: 15د / 5د / 3د / 1د — انتهاء ثابت: 3 دقائق
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

NAMES = {
    1: "RSI(2) كونورز",
    2: "عودة داخل الباند %B",
    3: "ستوكاستك تقاطع متطرف",
    4: "شريط EMA ارتداد ترند",
    5: "دايفرجنس RSI",
    6: "إنغلفينغ عند الباند",
    7: "انضغاط كيلتنر كسر",
    8: "CCI متطرف عكسي",
    9: "ويليامز %R متطرف",
    10: "هايكين آشي 3 شمعات",
}

TFS = [("tf15", "15د"), ("tf5", "5د"), ("tf3", "3د"), ("tf1", "1د")]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Top10Small")

def fetch(sym, iv, period):
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(sym).history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(inplace=True)
            df = df[(df["High"]>=df["Low"]) & (df["Open"]>0) & (df["Close"]>0)]
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return None

def rsi_ser(c, p):
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50)

def atr_ser(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def build_tf(df, tf_min):
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    cs = df["Close"]; hs = df["High"]; ls = df["Low"]
    d = {"times": df.index, "tf": tf_min, "o": o, "h": h, "l": l, "c": c}
    bb_mid = cs.rolling(20).mean()
    bb_sd = cs.rolling(20).std()
    d["bb_up"] = (bb_mid + 2*bb_sd).to_numpy(dtype=float)
    d["bb_lo"] = (bb_mid - 2*bb_sd).to_numpy(dtype=float)
    d["rsi14"] = rsi_ser(cs, 14).to_numpy(dtype=float)
    d["rsi2"] = rsi_ser(cs, 2).to_numpy(dtype=float)
    ll = ls.rolling(14).min()
    hh = hs.rolling(14).max()
    k = 100 * (cs - ll) / (hh - ll).replace(0, np.nan)
    d["stk"] = k.to_numpy(dtype=float)
    d["std_"] = k.rolling(3).mean().to_numpy(dtype=float)
    d["willr"] = (-100 * (hh - cs) / (hh - ll).replace(0, np.nan)).to_numpy(dtype=float)
    tp = (hs + ls + cs) / 3.0
    ma = tp.rolling(20).mean()
    md = (tp - ma).abs().rolling(20).mean()
    d["cci"] = ((tp - ma) / (0.015 * md.replace(0, np.nan))).to_numpy(dtype=float)
    e20 = cs.ewm(span=20, adjust=False).mean()
    at = atr_ser(hs, ls, cs, 10)
    ku = e20 + 2*at
    kl = e20 - 2*at
    d["ku"] = ku.to_numpy(dtype=float)
    d["kl"] = kl.to_numpy(dtype=float)
    d["sqz"] = ((bb_mid + 2*bb_sd < ku) & (bb_mid - 2*bb_sd > kl)).to_numpy(dtype=bool)
    d["e20"] = e20.to_numpy(dtype=float)
    d["e50"] = cs.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)
    d["e100"] = cs.ewm(span=100, adjust=False).mean().to_numpy(dtype=float)
    ha_c = ((df["Open"]+df["High"]+df["Low"]+df["Close"])/4).to_numpy(dtype=float)
    n = len(ha_c)
    ha_o = np.zeros(n)
    if n > 0:
        ha_o[0] = (o[0] + c[0]) / 2
        for i in range(1, n):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
    d["ha_o"] = ha_o
    d["ha_c"] = ha_c
    return d

def prepare(sym):
    d3 = fetch(sym, "3m", f"{HISTORY_DAYS}d")
    d5 = fetch(sym, "5m", f"{HISTORY_DAYS}d")
    d15 = fetch(sym, "15m", f"{HISTORY_DAYS}d")
    d60 = fetch(sym, "1h", f"{HISTORY_DAYS}d")
    if d3 is None or d5 is None or d15 is None or d60 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d3 = d3[d3.index + pd.Timedelta(minutes=3) <= now]
    d5 = d5[d5.index + pd.Timedelta(minutes=5) <= now]
    d15 = d15[d15.index + pd.Timedelta(minutes=15) <= now]
    d60 = d60[d60.index + pd.Timedelta(hours=1) <= now]
    if len(d3) < 500 or len(d15) < 200:
        return None
    return {
        "c3map": d3["Close"].to_dict(),
        "tf15": build_tf(d15, 15),
        "tf5": build_tf(d5, 5),
        "tf3": build_tf(d3, 3),
        "tf1": build_tf(d60.resample("1min", label="left", closed="left").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}).dropna(), 1) if len(d60) > 100 else None,
    }

def emit(trades, times, i, c, dr, c3map, tf_min):
    T = times[i] + pd.Timedelta(minutes=3)
    exit_px = c3map.get(T)
    if exit_px is None or (isinstance(exit_px, float) and np.isnan(exit_px)):
        return
    entry = c[i]
    if dr == "CALL":
        win = exit_px > entry
    else:
        win = exit_px < entry
    trades.append((times[i], bool(win)))

def evaluate(data, sid, tfk):
    c3map = data["c3map"]
    D = data.get(tfk)
    if D is None:
        return []
    t = D["times"]; o = D["o"]; h = D["h"]; l = D["l"]; c = D["c"]
    tf = D["tf"]
    n = len(c)
    trades = []
    start = 110 if n > 150 else 40

    if sid == 1:
        r2 = D["rsi2"]
        for i in range(start, n):
            if r2[i] <= 10:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif r2[i] >= 90:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 2:
        bu = D["bb_up"]; bl = D["bb_lo"]
        for i in range(start, n):
            if c[i-1] < bl[i-1] and c[i] > bl[i]:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif c[i-1] > bu[i-1] and c[i] < bu[i]:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 3:
        k = D["stk"]; kd = D["std_"]
        for i in range(start, n):
            if np.isnan(k[i]) or np.isnan(kd[i]) or np.isnan(k[i-1]) or np.isnan(kd[i-1]):
                continue
            if k[i-1] < kd[i-1] and k[i] > kd[i] and k[i] < 30:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif k[i-1] > kd[i-1] and k[i] < kd[i] and k[i] > 70:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 4:
        e20 = D["e20"]; e50 = D["e50"]; e100 = D["e100"]
        for i in range(start, n):
            if e20[i] > e50[i] > e100[i] and l[i] <= e20[i] and c[i] > e20[i] and c[i] > o[i]:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif e20[i] < e50[i] < e100[i] and h[i] >= e20[i] and c[i] < e20[i] and c[i] < o[i]:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 5:
        r14 = D["rsi14"]
        for i in range(start, n):
            a = max(0, i-15)
            b = i-5
            if b <= a:
                continue
            p_lo = a + int(np.argmin(l[a:b]))
            p_hi = a + int(np.argmax(h[a:b]))
            if l[i] < l[p_lo] and r14[i] > r14[p_lo] and l[i] <= l[i-4:i+1].min():
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif h[i] > h[p_hi] and r14[i] < r14[p_hi] and h[i] >= h[i-4:i+1].max():
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 6:
        bu = D["bb_up"]; bl = D["bb_lo"]
        for i in range(start, n):
            eng_c = (c[i] > o[i] and c[i-1] < o[i-1] and c[i] >= o[i-1] and o[i] <= c[i-1])
            eng_p = (c[i] < o[i] and c[i-1] > o[i-1] and c[i] <= o[i-1] and o[i] >= c[i-1])
            if eng_c and c[i] < bl[i]:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif eng_p and c[i] > bu[i]:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 7:
        ku = D["ku"]; kl = D["kl"]; sq = D["sqz"]
        for i in range(start, n):
            if not sq[i-1]:
                continue
            if c[i] > ku[i]:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif c[i] < kl[i]:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 8:
        cci = D["cci"]
        for i in range(start, n):
            if np.isnan(cci[i]):
                continue
            if cci[i] <= -200:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif cci[i] >= 200:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 9:
        wr = D["willr"]
        for i in range(start, n):
            if np.isnan(wr[i]):
                continue
            if wr[i] <= -90:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif wr[i] >= -10:
                emit(trades, t, i, c, "PUT", c3map, tf)
    elif sid == 10:
        ho = D["ha_o"]; hc = D["ha_c"]
        for i in range(start, n):
            red3 = (hc[i-1] < ho[i-1]) and (hc[i-2] < ho[i-2]) and (hc[i-3] < ho[i-3])
            grn3 = (hc[i-1] > ho[i-1]) and (hc[i-2] > ho[i-2]) and (hc[i-3] > ho[i-3])
            if red3 and hc[i] > ho[i]:
                emit(trades, t, i, c, "CALL", c3map, tf)
            elif grn3 and hc[i] < ho[i]:
                emit(trades, t, i, c, "PUT", c3map, tf)
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
    log.info("بدء العشرة الكبار على الفريمات الصغيرة (انتهاء 3د)")
    start = time.time()

    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            log.info(f"{sym}: جاهز")
        except Exception as e:
            log.error(f"{sym}: {e}")
            data_by_sym[sym] = None
        time.sleep(0.5)

    rows = []
    for sid in range(1, 11):
        for tfk, tfl in TFS:
            trades = []
            for sym, data in data_by_sym.items():
                if data is None:
                    continue
                trades.extend(evaluate(data, sid, tfk))
            st = stats(trades)
            if not st:
                rows.append((sid, tfl, 0, 0.0, False, 0.0, 0.0, 0.0))
                continue
            rob, w1, w2 = robustness(trades)
            per_day = round(st["total"] / HISTORY_DAYS, 1)
            rows.append((sid, tfl, st["total"], st["wr"], rob, w1, w2, per_day))
            log.info(f"استراتيجية {sid} على {tfl}: {st['total']} صفقة {st['wr']}%")

    msg = f"⏱️ *العشرة الكبار — فريمات صغيرة*\n(5 أزواج × 60 يوماً × انتهاء 3د)\n\n"
    msg += "```\n"
    msg += f"{'#':<4}{'فريم':<6}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>4}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<4}{r[1]:<6}{r[2]:>7}{r[3]:>7.1f}%{r[7]:>6}{mark:>4}\n"
    msg += "```\n\n"
    msg += f"📋 الاستراتيجيات:\n"
    for sid in NAMES:
        msg += f"{sid}) {NAMES[sid]}\n"

    msg += f"\n🧪 صلابة (نصف|نصف) للأهم:\n"
    shown = 0
    for r in rows:
        if r[2] >= 200 and shown < 12:
            msg += f"• {r[0]}/{r[1]}: {r[5]:.1f}% | {r[6]:.1f}%\n"
            shown += 1

    valid = [r for r in rows if r[2] >= 300]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += f"• أفضل تركيبة: *{best[3]}%* (استراتيجية {best[0]} على {best[1]}، {best[2]} صفقة)\n"
        cands = [r for r in valid if r[4] and r[3] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *تتفوق على H2 بـ2+ وصلبة:*\n"
            for r in cands:
                msg += f"• استراتيجية {r[0]} على {r[1]}: *{r[3]}%*\n"
            msg += f"\n⏳ H2 تبقى على الديمو — الفائزة تنتظر ديمو خاصة\n"
        else:
            msg += f"\n✅ *لا تفوق على H2* — الفريمات الصغيرة لا تضيف حافة\n"
    else:
        msg += f"\n⚠️ صفقات قليلة (<300) — استرشادي فقط\n"

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
