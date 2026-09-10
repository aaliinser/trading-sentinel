#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — فرضيات ثنائية جديدة (VERSION: 1=نطاق، 2=تشبع، 3=ترند طويل)
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

VERSION = int(os.getenv("VERSION", "1"))

SYMBOLS = [
    "USDJPY=X","AUDJPY=X","EURJPY=X","EURUSD=X","GBPUSD=X",
    "EURGBP=X","CADJPY=X","EURCAD=X","GBPCAD=X","AUDCHF=X",
    "AUDUSD=X","USDCHF=X","CHFJPY=X","AUDCAD=X","USDCAD=X",
    "EURAUD=X","EURCHF=X","GBPJPY=X","GBPCHF=X","GBPAUD=X"
]

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
log = logging.getLogger("BinaryHyp")

def fetch(sym, iv, period):
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(sym).history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for c in ["Open","High","Low","Close"]:
                if c not in df.columns:
                    df[c] = np.nan
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

def add_ind(df):
    if df is None or len(df) < 120:
        return None
    df = df.copy()
    c = df["Close"]
    df["EMA35"] = c.ewm(span=35, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/14, min_periods=14).mean()
    al = l.ewm(alpha=1/14, min_periods=14).mean()
    df["RSI"] = (100 - (100 / (1 + ag / al.replace(0, np.nan)))).fillna(50)
    pc = c.shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, min_periods=14).mean()
    um = df["High"].diff()
    dm = -df["Low"].diff()
    pdm = pd.Series(np.where((um>dm)&(um>0), um, 0.0), index=df.index)
    mdm = pd.Series(np.where((dm>um)&(dm>0), dm, 0.0), index=df.index)
    pdi = 100*pdm.ewm(alpha=1/14).mean()/df["ATR"].replace(0, np.nan)
    mdi = 100*mdm.ewm(alpha=1/14).mean()/df["ATR"].replace(0, np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1/14).mean()
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["BBU"] = mid + 2*sd
    df["BBL"] = mid - 2*sd
    df["RT"] = df["High"].rolling(48).max().shift(1)
    df["RB"] = df["Low"].rolling(48).min().shift(1)
    df["UW"] = df["High"] - df[["Open","Close"]].max(axis=1)
    df["LW"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    return df

def collect_v1(df):
    """ارتداد أطراف النطاق في سوق عرضي (15m، انتهاء 45د)"""
    trades = []
    for i in range(60, len(df)-4):
        r = df.iloc[i]
        adx = r.get("ADX")
        atr = r.get("ATR")
        rt = r.get("RT")
        rb = r.get("RB")
        if pd.isna(adx) or pd.isna(atr) or pd.isna(rt) or pd.isna(rb):
            continue
        if float(adx) >= 20.0:
            continue
        body = abs(float(r["Close"]) - float(r["Open"]))
        if body <= 0:
            continue
        if float(r["High"]) >= float(rt) - 0.2*float(atr):
            if float(r["Close"]) < float(r["Open"]) and float(r["UW"]) >= body:
                ex = float(df.iloc[i+3]["Close"])
                en = float(r["Close"])
                trades.append({"time": df.index[i], "dr": "PUT", "entry": en, "exit": ex})
        if float(r["Low"]) <= float(rb) + 0.2*float(atr):
            if float(r["Close"]) > float(r["Open"]) and float(r["LW"]) >= body:
                ex = float(df.iloc[i+3]["Close"])
                en = float(r["Close"])
                trades.append({"time": df.index[i], "dr": "CALL", "entry": en, "exit": ex})
    return trades

def collect_v2(df):
    """تطرف RSI + بولينجر (5m، انتهاء 15د)"""
    trades = []
    for i in range(60, len(df)-4):
        r = df.iloc[i]
        rsi = r.get("RSI")
        bbu = r.get("BBU")
        bbl = r.get("BBL")
        if pd.isna(rsi) or pd.isna(bbu) or pd.isna(bbl):
            continue
        en = float(r["Close"])
        if float(rsi) >= 75.0 and en >= float(bbu):
            ex = float(df.iloc[i+3]["Close"])
            trades.append({"time": df.index[i], "dr": "PUT", "entry": en, "exit": ex})
        if float(rsi) <= 25.0 and en <= float(bbl):
            ex = float(df.iloc[i+3]["Close"])
            trades.append({"time": df.index[i], "dr": "CALL", "entry": en, "exit": ex})
    return trades

def collect_v3(df):
    """ترند ساعة + انتهاء 4 ساعات"""
    trades = []
    for i in range(80, len(df)-5):
        r = df.iloc[i]
        e35 = r.get("EMA35")
        e50 = r.get("EMA50")
        if pd.isna(e35) or pd.isna(e50):
            continue
        en = float(r["Close"])
        if float(e35) > float(e50) and en > float(e35):
            if float(r["Low"]) <= float(e50) and float(r["Close"]) > float(r["Open"]):
                ex = float(df.iloc[i+4]["Close"])
                trades.append({"time": df.index[i], "dr": "CALL", "entry": en, "exit": ex})
        if float(e35) < float(e50) and en < float(e35):
            if float(r["High"]) >= float(e50) and float(r["Close"]) < float(r["Open"]):
                ex = float(df.iloc[i+4]["Close"])
                trades.append({"time": df.index[i], "dr": "PUT", "entry": en, "exit": ex})
    return trades

def stats(trades):
    if not trades:
        return None
    wins = 0
    for t in trades:
        if t["dr"] == "CALL":
            if t["exit"] > t["entry"]:
                wins += 1
        else:
            if t["exit"] < t["entry"]:
                wins += 1
    total = len(trades)
    wr = round(100 * wins / total, 2)
    pnl = round(wins * STAKE * PAYOUT - (total - wins) * STAKE, 2)
    return {"total": total, "wins": wins, "wr": wr, "pnl": pnl}

def robustness(trades):
    if len(trades) < 60:
        return False, 0, 0
    mid = len(trades) // 2
    s1 = stats(trades[:mid])
    s2 = stats(trades[mid:])
    if not s1 or not s2:
        return False, 0, 0
    ok = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return ok, s1["wr"], s2["wr"]

def fmt_sym(s):
    b = s.replace("=X", "")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def build_report():
    names = {1: "ارتداد أطراف النطاق (15م/45د)", 2: "تشبع RSI+بولينجر (5م/15د)", 3: "ترند ساعة (1س/4س)"}
    if VERSION not in names:
        return f"ℹ️ النسخة {VERSION} غير مستخدمة بهذه الجولة"
    log.info(f"بدء الفرضية {VERSION}: {names[VERSION]}")
    start = time.time()

    if VERSION == 1:
        iv, period = "15m", "730d"
    elif VERSION == 2:
        iv, period = "5m", "60d"
    else:
        iv, period = "1h", "730d"

    all_trades = []
    sym_stats = []
    for sym in SYMBOLS:
        try:
            df = fetch(sym, iv, period)
            df = add_ind(df)
            if df is None:
                continue
            if VERSION == 1:
                tr = collect_v1(df)
            elif VERSION == 2:
                tr = collect_v2(df)
            else:
                tr = collect_v3(df)
            for t in tr:
                t["sym"] = sym
            all_trades.extend(tr)
            s = stats(tr)
            if s and s["total"] >= 10:
                sym_stats.append((sym, s))
            log.info(f"{sym}: {len(tr)} صفقة")
            time.sleep(1)
        except Exception as e:
            log.error(f"{sym}: {e}")

    overall = stats(all_trades)
    if not overall:
        return f"❌ الفرضية {VERSION}: لا صفقات"

    robust, wr1, wr2 = robustness(all_trades)
    sym_stats.sort(key=lambda x: x[1]["wr"], reverse=True)

    msg = f"🎯 *فرضية {VERSION}: {names[VERSION]}*\n"
    msg += f"• صفقات: *{overall['total']}*\n"
    msg += f"• فوز: *{overall['wr']}%*\n"
    msg += f"• صافي: *{overall['pnl']:+.2f}$*\n"
    msg += f"• التعادل: {BREAKEVEN}%\n\n"
    msg += f"📈 أفضل 5:\n"
    for idx in range(min(5, len(sym_stats))):
        sym, s = sym_stats[idx]
        msg += f"• {fmt_sym(sym)}: {s['wr']}% ({s['total']})\n"
    msg += f"\n🧪 صلابة: "
    if robust:
        msg += f"✅ ({wr1}%|{wr2}%)\n"
    else:
        msg += f"❌ ({wr1}%|{wr2}%)\n"
    msg += f"\n💡 الحكم: "
    if robust and overall["wr"] >= 55:
        msg += f"🟢 رابحة صلبة - نعتمدها\n"
    elif robust and overall["wr"] >= BREAKEVEN:
        msg += f"🟡 هامشية صلبة\n"
    elif overall["wr"] >= BREAKEVEN:
        msg += f"🟠 فوق التعادل غير صلبة\n"
    else:
        msg += f"🔴 مرفوضة\n"
    if overall["total"] < 100:
        msg += f"⚠️ صفقات قليلة\n"
    msg += f"\n⏱️ {time.time()-start:.0f}ث"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
        requests.post(url, json=payload, timeout=15)
        log.info("✅ أُرسل")
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
