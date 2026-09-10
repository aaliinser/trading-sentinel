#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
غيث v20 — باك تست كامل ومصلح
=====================================================================
"""
import os, sys, time, logging
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
ADX_P = 14
MAX_DIST_EMA = 2.0
MIN_SPACE = 0.3
LVL_PROX = 0.6
MAX_DEV = 0.0010
MAX_AHEAD = 0.0004
TOUCH_TOL = 0.0003
REJ_BODY = 0.35
MIN_SCORE = 2
MAX_SC = 4
ADX_M15 = 18.0
ADX_H1 = 20.0
RSI_C_MIN = 38.0
RSI_C_MAX = 62.0
RSI_P_MIN = 38.0
RSI_P_MAX = 62.0
RN_LARGE = 0.5
RN_SMALL = 0.005
MIN_R = 200
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Backtest_v20")

def fetch(sym, iv, period):
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
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return None

def add_indicators(df):
    if df is None or df.empty or len(df) < MIN_R:
        return df
    df = df.copy()
    if HAS_TA:
        df["EMA_35"] = ta.ema(df["Close"], length=EMA_F)
        df["EMA_50"] = ta.ema(df["Close"], length=EMA_S)
        df["RSI"] = ta.rsi(df["Close"], length=RSI_P)
        m = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if m is not None:
            df = pd.concat([df, m], axis=1)
            df.rename(columns={"MACD_12_26_9":"MACD","MACDh_12_26_9":"MACD_HIST","MACDs_12_26_9":"MACD_SIGNAL"}, inplace=True)
        a = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_P)
        if a is not None:
            df["ATR"] = a
        x = ta.adx(df["High"], df["Low"], df["Close"], length=ADX_P)
        if x is not None:
            for c in x.columns:
                df[c] = x[c]
            df.rename(columns={f"ADX_{ADX_P}":"ADX", f"DMP_{ADX_P}":"PLUS_DI", f"DMN_{ADX_P}":"MINUS_DI"}, inplace=True)
    else:
        df["EMA_35"] = df["Close"].ewm(span=EMA_F, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=EMA_S, adjust=False).mean()
        d = df["Close"].diff()
        g = d.clip(lower=0)
        l = -d.clip(upper=0)
        ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
        al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
        df["RSI"] = (100 - (100 / (1 + ag / al.replace(0, np.nan)))).fillna(50)
        e12 = df["Close"].ewm(span=12, adjust=False).mean()
        e26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = e12 - e26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
        pc = df["Close"].shift(1)
        tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1/ATR_P, min_periods=ATR_P).mean()
        um = df["High"].diff()
        dm = -df["Low"].diff()
        pdm = pd.Series(np.where((um>dm)&(um>0), um, 0.0), index=df.index)
        mdm = pd.Series(np.where((dm>um)&(dm>0), dm, 0.0), index=df.index)
        pdi = 100*pdm.ewm(alpha=1/ADX_P).mean()/df["ATR"].replace(0, np.nan)
        mdi = 100*mdm.ewm(alpha=1/ADX_P).mean()/df["ATR"].replace(0, np.nan)
        dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
        df["ADX"] = dx.ewm(alpha=1/ADX_P).mean()
        df["PLUS_DI"] = pdi
        df["MINUS_DI"] = mdi
    df["RS"] = df["Low"].rolling(LVL_LB, min_periods=20).min()
    df["RR"] = df["High"].rolling(LVL_LB, min_periods=20).max()
    df["H20"] = df["High"].rolling(20, min_periods=10).max()
    df["L20"] = df["Low"].rolling(20, min_periods=10).min()
    df["BODY"] = (df["Close"] - df["Open"]).abs()
    df["RANGE"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["UWICK"] = df["High"] - df[["Open","Close"]].max(axis=1)
    df["LWICK"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    return df

def _v(*vs):
    for v in vs:
        if v is None:
            return False
        try:
            if pd.isna(v) or not np.isfinite(float(v)):
                return False
        except Exception:
            return False
    return True

def ema_distance_ok(last):
    c = float(last["Close"])
    e = float(last["EMA_35"])
    a = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
    return a > 0 and abs(c - e) <= MAX_DIST_EMA * a

def trend_direction(last_15, prev_15, last_h1):
    if not _v(last_15["EMA_35"], last_15["EMA_50"], prev_15["EMA_35"],
              last_h1["EMA_35"], last_h1["EMA_50"], last_15["Close"], last_h1["Close"]):
        return None
    hb = last_h1["Close"] > last_h1["EMA_35"] > last_h1["EMA_50"]
    hr = last_h1["Close"] < last_h1["EMA_35"] < last_h1["EMA_50"]
    mb = last_15["Close"] > last_15["EMA_35"] > last_15["EMA_50"] and last_15["EMA_35"] > prev_15["EMA_35"]
    mr = last_15["Close"] < last_15["EMA_35"] < last_15["EMA_50"] and last_15["EMA_35"] < prev_15["EMA_35"]
    if hb and mb:
        return "CALL"
    if hr and mr:
        return "PUT"
    return None

def space_ok(last, dr):
    if not _v(last.get("ATR"), last.get("H20"), last.get("L20"), last.get("Close")):
        return True
    a = float(last["ATR"])
    c = float(last["Close"])
    if a <= 0:
        return True
    h = float(last["H20"])
    l = float(last["L20"])
    ms = MIN_SPACE * a
    if dr == "CALL":
        return (h - c) >= ms
    return (c - l) >= ms

def find_level(last, dr):
    c = float(last["Close"])
    a = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
    if a <= 0:
        return None
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

def score(last, prev, h1, lv):
    sc = {"T":0, "M":0, "L":0, "Q":0}
    if _v(last["EMA_35"], last["EMA_50"], prev["EMA_35"], h1["EMA_35"], h1["EMA_50"], last["Close"], h1["Close"]):
        hb = h1["Close"] > h1["EMA_35"] > h1["EMA_50"]
        hr = h1["Close"] < h1["EMA_35"] < h1["EMA_50"]
        mb = last["Close"] > last["EMA_35"] > last["EMA_50"] and last["EMA_35"] > prev["EMA_35"]
        mr = last["Close"] < last["EMA_35"] < last["EMA_50"] and last["EMA_35"] < prev["EMA_35"]
        if (hb and mb) or (hr and mr):
            sc["T"] = 1
    if _v(last["RSI"], prev["RSI"]):
        r = float(last["RSI"])
        pr = float(prev["RSI"])
        h = float(last["MACD_HIST"]) if pd.notna(last.get("MACD_HIST")) else 0.0
        ph = float(prev["MACD_HIST"]) if pd.notna(prev.get("MACD_HIST")) else 0.0
        bb = RSI_C_MIN <= r <= RSI_C_MAX and r > pr
        br = RSI_P_MIN <= r <= RSI_P_MAX and r < pr
        if (bb and h > 0 and h >= ph) or (br and h < 0 and h <= ph):
            sc["M"] = 1
    sc["L"] = 1 if lv is not None else 0
    if _v(last.get("ADX"), h1.get("ADX")):
        am = float(last["ADX"])
        ah = float(h1["ADX"])
        if am >= ADX_M15 and ah >= ADX_H1:
            sc["Q"] = 1
    return sc

def touch(rej, level, dr):
    t = TOUCH_TOL * float(rej["Close"])
    if dr == "CALL":
        return float(rej["Low"]) <= level + t
    return float(rej["High"]) >= level - t

def rejection(rej, prev, level, dr):
    close = float(rej["Close"])
    body = float(abs(rej["Close"] - rej["Open"]))
    fr = float(rej["High"] - rej["Low"])
    if fr <= 0 or body <= 0:
        return False
    br = body / fr
    brej = br >= REJ_BODY
    if dr == "CALL":
        lw = float(rej.get("LWICK", 0)) if pd.notna(rej.get("LWICK")) else 0.0
        pin = lw >= 0.6 * fr and br <= 0.4
        eng = (rej["Close"] > rej["Open"] and
               prev["Close"] < prev["Open"] and
               rej["Close"] >= prev["Open"] and
               rej["Open"] <= prev["Close"])
        return (brej or pin or eng) and close > level
    uw = float(rej.get("UWICK", 0)) if pd.notna(rej.get("UWICK")) else 0.0
    pin = uw >= 0.6 * fr and br <= 0.4
    eng = (rej["Close"] < rej["Open"] and
           prev["Close"] > prev["Open"] and
           rej["Close"] <= prev["Open"] and
           rej["Open"] >= prev["Close"])
    return (brej or pin or eng) and close < level

def deviation_ok(level, entry, dr):
    dn = (level - entry) / entry
    up = (entry - level) / entry
    if dr == "PUT":
        if dn > MAX_DEV:
            return False
        if up > MAX_AHEAD:
            return False
    else:
        if up > MAX_DEV:
            return False
        if dn > MAX_AHEAD:
            return False
    return True

def rsi_ok(rsi, dr):
    if rsi is None or pd.isna(rsi):
        return True
    r = float(rsi)
    if dr == "CALL":
        return r <= RSI_C_MAX + 5
    return r >= RSI_P_MIN - 5

def collect_candidates(sym):
    log.info(f"=== {sym} ===")
    d15 = fetch(sym, SCAN_TF, f"{HISTORY_DAYS}d")
    d5 = fetch(sym, SNIPER_TF, "60d")
    d1h = fetch(sym, TREND_TF, f"{HISTORY_DAYS}d")
    if d15 is None or d5 is None or d1h is None:
        return []
    d15 = add_indicators(d15)
    d5 = add_indicators(d5)
    d1h = add_indicators(d1h)
    if d15 is None or d5 is None or d1h is None:
        return []
    if len(d15) < MIN_R or len(d5) < MIN_R or len(d1h) < 100:
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

        if not ema_distance_ok(last_15):
            continue
        dr = trend_direction(last_15, prev_15, last_h1)
        if dr is None:
            continue
        if not space_ok(last_15, dr):
            continue
        level = find_level(last_15, dr)
        if level is None:
            continue
        sc = score(last_15, prev_15, last_h1, level)
        if sum(sc.values()) < MIN_SCORE:
            continue

        d5_view = d5.iloc[max(0, i-20):i+1]
        if len(d5_view) < 10:
            continue
        conf = d5_view.iloc[-1]
        rej = d5_view.iloc[-2]
        prev = d5_view.iloc[-3]

        if not touch(rej, level, dr):
            continue
        if not rejection(rej, prev, level, dr):
            continue
        rej_close = float(rej["Close"])
        conf_close = float(conf["Close"])
        if dr == "CALL" and conf_close < rej_close:
            continue
        if dr == "PUT" and conf_close > rej_close:
            continue
        if not deviation_ok(level, conf_close, dr):
            continue
        rsi_15 = last_15.get("RSI")
        if not rsi_ok(rsi_15, dr):
            continue
        if not (7 <= cur_time.hour < 21):
            continue

        end_idx = i + 3
        if end_idx >= len(d5):
            continue
        exit_price = float(d5.iloc[end_idx]["Close"])

        candidates.append({
            "time": cur_time,
            "sym": sym,
            "dr": dr,
            "entry": conf_close,
            "exit": exit_price,
        })
    log.info(f"{sym}: {len(candidates)} صفقة")
    return candidates

def stats(candidates):
    if not candidates:
        return None
    wins = 0
    for c in candidates:
        if c["dr"] == "CALL":
            if c["exit"] > c["entry"]:
                wins += 1
        else:
            if c["exit"] < c["entry"]:
                wins += 1
    total = len(candidates)
    wr = round(100 * wins / total, 2)
    pnl = round(wins * STAKE * PAYOUT - (total - wins) * STAKE, 2)
    return {"total": total, "wins": wins, "wr": wr, "pnl": pnl}

def robustness(candidates):
    if len(candidates) < 60:
        return False, 0, 0
    mid = len(candidates) // 2
    s1 = stats(candidates[:mid])
    s2 = stats(candidates[mid:])
    if not s1 or not s2:
        return False, 0, 0
    robust = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return robust, s1["wr"], s2["wr"]

def fmt_sym(s):
    b = s.replace("=X", "")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def build_report():
    log.info(f"بدء باك تست v20 ({len(SYMBOLS)} زوجاً × {HISTORY_DAYS} يوم)")
    start = time.time()

    all_cands = {}
    for sym in SYMBOLS:
        try:
            all_cands[sym] = collect_candidates(sym)
            time.sleep(1)
        except Exception as e:
            log.error(f"{sym}: {e}")
            all_cands[sym] = []

    all_flat = []
    sym_stats = []
    for sym, cands in all_cands.items():
        s = stats(cands)
        if s:
            sym_stats.append((sym, s))
            all_flat.extend(cands)

    overall = stats(all_flat)
    if not overall:
        return "❌ لا توجد صفقات كافية"

    robust, wr1, wr2 = robustness(all_flat)
    sym_stats.sort(key=lambda x: x[1]["wr"], reverse=True)

    msg = f"🏗️ *غيث v20 — باك تست*\n"
    msg += f"({len(SYMBOLS)} زوجاً × {HISTORY_DAYS} يوم)\n\n"

    msg += f"🎯 *الأرقام الأساسية:*\n"
    msg += f"• عدد الصفقات: *{overall['total']}*\n"
    msg += f"• نسبة الفوز: *{overall['wr']}%*\n"
    msg += f"• صافي الربح: *{overall['pnl']:+.2f}$*\n"
    msg += f"• (نقطة التعادل = {BREAKEVEN}%)\n\n"

    msg += f"📈 *الأزواج مرتبة:*\n"
    msg += f"```\n"
    msg += f"{'الزوج':<10} {'#':>5} {'WR':>7} {'صافي':>10}\n"
    msg += f"{'-'*10} {'-'*5} {'-'*7} {'-'*10}\n"
    for idx, (sym, s) in enumerate(sym_stats):
        if s["total"] >= 10:
            medal = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else " "))
            msg += f"{medal}{fmt_sym(sym):<10} {s['total']:>5} {s['wr']:>6.1f}% {s['pnl']:>+9.0f}$\n"
    msg += f"```\n\n"

    msg += f"🧪 *اختبار الصلابة (نصفان):*\n"
    if robust:
        msg += f"✅ *صلبة* — النصف 1: {wr1}% | النصف 2: {wr2}%\n"
    else:
        msg += f"❌ *غير صلبة* — النصف 1: {wr1}% | النصف 2: {wr2}%\n"

    msg += f"\n💡 *الحكم:*\n"
    if robust and overall["wr"] >= 55:
        msg += f"🟢 *رابحة صلبة* — نعتمدها\n"
    elif robust and overall["wr"] >= BREAKEVEN:
        msg += f"🟡 *هامشية لكن صلبة* — اعتماد بحذر\n"
    elif overall["wr"] >= BREAKEVEN:
        msg += f"🟠 *فوق التعادل لكن غير صلبة*\n"
    else:
        msg += f"🔴 *تحت التعادل — مرفوضة*\n"

    msg += f"\n⏱️ انتهى في {time.time()-start:.0f} ثانية"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
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
