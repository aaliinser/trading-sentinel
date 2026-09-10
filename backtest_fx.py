#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — بوت إشارات حي: تشبع RSI + بولينجر (5m) — انتهاء 15 دقيقة
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

SYMBOLS = os.getenv("SYMBOLS_H2", "USDJPY=X,EURAUD=X,USDCHF=X,EURCAD=X,CADJPY=X").split(",")
RSI_P = 14
BB_P = 20
BB_K = 2.0
RSI_HI = 75.0
RSI_LO = 25.0
EXPIRY_MIN = 15
MAX_ALERTS_DAY = int(os.getenv("MAX_ALERTS_DAY", "8"))
MAX_TRADES_DAY = int(os.getenv("MAX_TRADES_DAY", "5"))
STOP_AFTER_LOSSES = 3
STOP_HOURS = 4
STATE_FILE = "state_h2.json"

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("H2")

def load_state():
    p = Path(STATE_FILE)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
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
        d = {"date": today, "alerts": 0, "trades": 0, "wins": 0, "losses": 0, "cl": 0}
        st["day"] = d
    return d

def tg_send(text, reply_to=None):
    if not TG_TOKEN or not TG_CHAT:
        log.info("TG disabled")
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    p = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
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

def fetch5m(sym):
    for a in range(3):
        try:
            df = yf.Ticker(sym).history(period="7d", interval="5m", auto_adjust=False, actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
            now = pd.Timestamp.now(tz="UTC")
            df = df[df.index <= now]
            if not df.empty and df.index[-1] + pd.Timedelta(minutes=5) > now:
                df = df.iloc[:-1]
            return df
        except Exception as e:
            log.warning(f"{sym} fetch {a}: {e}")
            time.sleep(2*a+1)
    return None

def indicators(df):
    c = df["Close"]
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    rsi = (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50)
    mid = c.rolling(BB_P).mean()
    sd = c.rolling(BB_P).std()
    return rsi, mid + BB_K*sd, mid - BB_K*sd, sd

def fmt_sym(s):
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b)==6 else s

def fmt_px(v):
    return f"{v:.3f}" if v > 50 else f"{v:.5f}"

def listen(st):
    if not TG_TOKEN:
        return
    try:
        off = st.get("tg_offset", 0)
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", params={"offset": off, "timeout": 0}, timeout=15)
        for u in r.json().get("result", []):
            uid = u.get("update_id", 0)
            if uid >= off:
                off = uid + 1
            m = u.get("message") or u.get("edited_message")
            if not m:
                continue
            t = (m.get("text") or "").lower()
            win = None
            if any(w in t for w in ["ربحت","رابحة","won","win"]):
                win = True
            elif any(w in t for w in ["خسرت","خاسرة","lost","lose"]):
                win = False
            if win is None:
                continue
            rt = m.get("reply_to_message") or {}
            mid = str(rt.get("message_id",""))
            op = st.get("open", {}).get(mid)
            d = day_obj(st)
            if op is None:
                tg_send("⚠️ رد على رسالة الإشارة مباشرة (Reply) لأسجل النتيجة")
                continue
            del st["open"][mid]
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
            pnl = round(d["wins"]*5.4 - d["losses"]*6, 2)
            tg_send(f"💰 سُجلت: {'✅' if win else '❌'}\n• اليوم: {d['trades']} صفقة | فوز {wr}%\n• صافي تقديري: {pnl:+.2f}$")
        st["tg_offset"] = off
    except Exception as e:
        log.warning(f"listen: {e}")

def scan(st):
    d = day_obj(st)
    if time.time() < st.get("stop_until", 0):
        log.info("موقوف مؤقتاً (خسائر متتالية)")
        return
    if d["alerts"] >= MAX_ALERTS_DAY:
        return
    if d["trades"] >= MAX_TRADES_DAY:
        return
    for sym in SYMBOLS:
        df = fetch5m(sym)
        if df is None or len(df) < 60:
            continue
        rsi, bbu, bbl, sd = indicators(df)
        i = len(df) - 1
        ct = df.index[i]
        key = ct.isoformat()
        lastmap = st.setdefault("last_sig", {})
        if lastmap.get(sym) == key:
            continue
        r = float(rsi.iloc[i])
        cl = float(df["Close"].iloc[i])
        bu = float(bbu.iloc[i])
        bl = float(bbl.iloc[i])
        sdv = float(sd.iloc[i])
        dr = None
        if r >= RSI_HI and cl >= bu:
            dr = "PUT"
        elif r <= RSI_LO and cl <= bl:
            dr = "CALL"
        if dr is None:
            continue
        lastmap[sym] = key
        d["alerts"] += 1
        over = abs(cl - (bu if dr == "PUT" else bl)) / sdv if sdv > 0 else 0.0
        exp = datetime.now(timezone.utc) + timedelta(minutes=EXPIRY_MIN)
        arrow = "🔴 PUT (هبوط)" if dr == "PUT" else "🟢 CALL (صعود)"
        txt = (f"🎯 *إشارة H2 — تشبع + بولينجر*\n\n"
               f"• الزوج: *{fmt_sym(sym)}*\n"
               f"• الاتجاه: {arrow}\n"
               f"• سعر الإشارة: {fmt_px(cl)}\n"
               f"• السبب: RSI {r:.1f} + إغلاق خارج الباند بمقدار {over:.1f}σ\n\n"
               f"⏱️ ادخل خلال 60 ثانية\n"
               f"⌛ الانتهاء: 15 دقيقة (حتى {exp.strftime('%H:%M')} UTC)\n"
               f"💰 شرط الـ payout: 85% فأعلى فقط\n\n"
               f"📊 تنبيهات اليوم: {d['alerts']}/{MAX_ALERTS_DAY}\n"
               f"📝 بعد الصفقة رد على الرسالة: ربحت / خسرت")
        mid_id = tg_send(txt)
        if mid_id:
            st.setdefault("open", {})[str(mid_id)] = {"sym": sym, "dr": dr, "px": cl}
        save_state(st)
        time.sleep(0.5)

def main():
    st = load_state()
    d = day_obj(st)
    if st.get("boot_date") != d["date"]:
        st["boot_date"] = d["date"]
        tg_send(f"🚀 بوت H2 بدأ\n• أزواج: {len(SYMBOLS)}\n• سقف تنبيهات: {MAX_ALERTS_DAY}/يوم\n• سقف صفقات: {MAX_TRADES_DAY}/يوم\n• توقف تلقائي: {STOP_AFTER_LOSSES} خسائر متتالية = {STOP_HOURS} ساعات")
    listen(st)
    scan(st)
    save_state(st)
    log.info("done")

if __name__ == "__main__":
    main()
