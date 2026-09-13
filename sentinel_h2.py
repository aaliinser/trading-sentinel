#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — بوت إشارات حي (v4 Pro)
RSI(14) Wilder + Bollinger(20, 2.0) ddof=0
+ مناطق دخول + سعر حي + إلحاح تلقائي
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

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("H2v4Pro")

# ─── State Management ───
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
        old_cl = d.get("cl", 0)
        d = {"date": today, "alerts": 0, "trades": 0, 
             "wins": 0, "losses": 0, "cl": old_cl}
        st["day"] = d
    return d

# ─── Telegram ───
def tg_send(text, reply_to=None):
    if not TG_TOKEN or not TG_CHAT:
        log.info("TG disabled")
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    p = {"chat_id": TG_CHAT, "text": text, 
         "parse_mode": "Markdown", 
         "disable_web_page_preview": True}
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

# ─── Data Fetching (3d window proven safe) ───
def fetch5m(sym):
    for a in range(3):
        try:
            df = yf.Ticker(sym).history(period="3d", interval="5m",
                                        auto_adjust=False, 
                                        actions=False, timeout=20)
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

# ─── Indicators (Wilder RSI + ddof=0 BB = TradingView exact) ───
def indicators(df):
    c = df["Close"]
    
    # Wilder's RSI (exact TradingView formula)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / RSI_P
    avg_gain = gain.ewm(alpha=alpha, adjust=False, 
                        min_periods=RSI_P).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, 
                        min_periods=RSI_P).mean()
    denom = avg_gain + avg_loss
    rsi = 100.0 * avg_gain / denom
    rsi = rsi.where(denom > 0, 50.0)
    
    # Bollinger Bands (ddof=0 = population std = TradingView)
    mid = c.rolling(BB_P, min_periods=BB_P).mean()
    sd = c.rolling(BB_P, min_periods=BB_P).std(ddof=0)
    bu = mid + BB_K * sd
    bl = mid - BB_K * sd
    
    return rsi, bu, bl, sd

# ─── Symbol Formatting ───
def fmt_sym(s):
    """USDJPY=X → USD/JPY (copyable in broker)"""
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def fmt_px(v):
    # Fix 3: Ensure standard float type for safe formatting
    v = float(v)
    return f"{v:.3f}" if v > 50 else f"{v:.5f}"

# ─── Urgency Logic ───
def calc_urgency(cl, band, sdv, direction):
    """
    Returns 4 values: (zone, urgency, emoji, text)
    zone: strong/mid/far
    urgency: enter_now / caution / skip
    """
    if sdv <= 0:
        return "unknown", "unknown", "⚠️", "خطأ في حساب الانحراف"
    
    if direction == "CALL":
        # Price should be at or below lower band (with 0.03% tolerance for live ticks)
        if cl <= band * 1.0003:
            zone = "strong"
            urgency = "enter_now"
            emoji = "✅"
            text = "ادخل فوراً خلال 60 ثانية\n(السعر عند الباند — فرصة مثالية)"
        elif cl <= band * 1.0015:
            zone = "mid"
            urgency = "enter_now"
            emoji = "⚡"
            text = "ادخل خلال 60 ثانية\n(السعر قريب من الباند)"
        else:
            zone = "far"
            urgency = "skip"
            emoji = "🚫"
            text = "تخطّ — السعر ابتعد عن الباند"
    else:  # PUT
        if cl >= band * 0.9997:
            zone = "strong"
            urgency = "enter_now"
            emoji = "✅"
            text = "ادخل فوراً خلال 60 ثانية\n(السعر عند الباند — فرصة مثالية)"
        elif cl >= band * 0.9985:
            zone = "mid"
            urgency = "enter_now"
            emoji = "⚡"
            text = "ادخل خلال 60 ثانية\n(السعر قريب من الباند)"
        else:
            zone = "far"
            urgency = "skip"
            emoji = "🚫"
            text = "تخطّ — السعر ابتعد عن الباند"
    
    return zone, urgency, emoji, text

# ─── Reports ───
def send_report(st):
    log_list = st.get("log", [])
    if not log_list:
        tg_send("📊 لا صفقات مسجلة بعد")
        return
    tot = len(log_list)
    wins = sum(1 for x in log_list if x["win"])
    v4_list = [x for x in log_list if x["v4"]]
    v4_tot = len(v4_list)
    v4_wins = sum(1 for x in v4_list if x["win"])
    bo_tot = tot - v4_tot
    bo_wins = wins - v4_wins
    wr = round(100 * wins / tot, 1) if tot else 0
    v4_wr = round(100 * v4_wins / v4_tot, 1) if v4_tot else 0
    bo_wr = round(100 * bo_wins / bo_tot, 1) if bo_tot else 0
    txt = (f"📊 *تقرير الديمو*\n\n"
           f"• كل الصفقات: *{tot}* | فوز *{wr}%*\n"
           f"• صفقات V4 ✅: *{v4_tot}* | فوز *{v4_wr}%*\n"
           f"• أساس فقط ➖: *{bo_tot}* | فوز *{bo_wr}%*\n\n"
           f"📌 V4 تُعتمد إذا تفوقت بنقطتين+ بعد 30+ صفقة")
    tg_send(txt)

def send_day_summary(st, old):
    tr = old.get("trades", 0)
    al = old.get("alerts", 0)
    if tr == 0 and al == 0:
        return
    w = old.get("wins", 0)
    ls = old.get("losses", 0)
    wr = round(100*w/tr, 1) if tr else 0
    pnl = round(w*5.4 - ls*6, 2)
    date = old.get("date", "?")
    logs = [x for x in st.get("log", []) if x.get("date") == date]
    v4 = [x for x in logs if x.get("v4")]
    v4w = sum(1 for x in v4 if x["win"])
    v4wr = round(100*v4w/len(v4), 1) if v4 else 0
    txt = (f"🌙 *ملخص يوم {date}*\n"
           f"• تنبيهات: {al} | صفقات: {tr}\n"
           f"• فوز {w} / خسارة {ls} | نسبة {wr}%\n"
           f"• صافي تقديري: {pnl:+.2f}$\n"
           f"• صفقات V4: {len(v4)} | نسبة {v4wr}%")
    tg_send(txt)

def send_week_summary(st):
    logs = st.get("log", [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    recent = []
    for x in logs:
        try:
            d = datetime.strptime(x.get("date",""), "%Y-%m-%d")
            d = d.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if d >= cutoff:
            recent.append(x)
    if not recent:
        return
    tot = len(recent)
    w = sum(1 for x in recent if x["win"])
    wr = round(100*w/tot, 1)
    v4 = [x for x in recent if x.get("v4")]
    v4w = sum(1 for x in v4 if x["win"])
    v4wr = round(100*v4w/len(v4), 1) if v4 else 0
    bo = tot - len(v4)
    bow = w - v4w
    bowr = round(100*bow/bo, 1) if bo else 0
    txt = (f"📅 *ملخص أسبوعي (آخر 7 أيام)*\n"
           f"• صفقات: {tot} | نسبة الفوز {wr}%\n"
           f"• V4: {len(v4)} صفقة | {v4wr}%\n"
           f"• أساس فقط: {bo} صفقة | {bowr}%\n\n"
           f"📌 بوابة الاعتماد: 50+ صفقة و≥53%")
    tg_send(txt)

# ─── Listen ───
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
            op = st.get("open", {}).get(mid)
            d = day_obj(st)
            if op is None:
                tg_send("⚠️ رد على رسالة الإشارة مباشرة")
                continue
            del st["open"][mid]
            st.setdefault("log", []).append({
                "date": d["date"],
                "sym": op.get("sym", "?"),
                "dr": op.get("dr", "?"),
                "v4": bool(op.get("v4", False)),
                "win": bool(win),
            })
            if len(st["log"]) > 600:
                st["log"] = st["log"][-600:]
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
            tg_send(f"💰 سُجلت: {'✅' if win else '❌'}\n"
                    f"• اليوم: {d['trades']} صفقة | فوز {wr}%\n"
                    f"• صافي تقديري: {pnl:+.2f}$")
        st["tg_offset"] = off
    except Exception as e:
        log.warning(f"listen: {e}")

# ─── Scan (with new message format) ───
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
        if np.isnan(sd.iloc[i]):
            continue
        
        r = float(rsi.iloc[i])
        cl = float(df["Close"].iloc[i])
        bu = float(bbu.iloc[i])
        bl = float(bbl.iloc[i])
        sdv = float(sd.iloc[i])
        
        dr = None
        band_ref = None
        if r >= RSI_HI and cl >= bu:
            dr = "PUT"
            band_ref = bu
        elif r <= RSI_LO and cl <= bl:
            dr = "CALL"
            band_ref = bl
        
        if dr is None:
            continue
        
        # V4 tag (0.5σ beyond band)
        over = abs(cl - band_ref) / sdv if sdv > 0 else 0.0
        v4 = over >= 0.5
        
        # Calculate entry zones
        if dr == "CALL":
            strong_entry = bl
            mid_entry = (cl + bl) / 2
        else:
            strong_entry = bu
            mid_entry = (cl + bu) / 2
        
        # Urgency check
        zone, urgency, emoji, urg_text = calc_urgency(cl, band_ref, sdv, dr)
        
        # Skip if too far from band
        if urgency == "skip":
            log.info(f"{sym}: إشارة ضعيفة — تخطي")
            lastmap[sym] = key
            continue
        
        # Expiry time
        exp = ct + timedelta(minutes=5 + EXPIRY_MIN)
        # Fix 2: Added missing red circle emoji for PUT
        arrow = "🔴 PUT (هبوط)" if dr == "PUT" else "🟢 CALL (صعود)"
        sym_copy = fmt_sym(sym)
        
        txt = (f"🎯 *إشارة H2 — تشبع + بولينجر*\n\n"
               f"📊 *الزوج:* `{sym_copy}`\n"
               f"📈 *الاتجاه:* {arrow}\n"
               f"💰 *السعر الحي الآن:* {fmt_px(cl)}\n\n"
               f"🎯 *مناطق الدخول:*\n"
               f"• قوية (عند الباند): {fmt_px(strong_entry)}\n"
               f"• وسطى (مقبولة): {fmt_px(mid_entry)}\n\n"
               f"⏱️ *التعليمات:*\n"
               f"{emoji} {urg_text}\n\n"
               f"📌 *السبب:* RSI {r:.1f} + السعر {'تحت' if dr=='CALL' else 'فوق'} الباند بـ {over:.1f}σ\n"
               f"🏷️ ضمن V4: {'✅' if v4 else '➖'}\n\n"
               f"⌛ الانتهاء: 15 دقيقة (حتى {exp.strftime('%H:%M')} UTC)\n"
               f"💰 شرط الـ payout: 85% فأعلى فقط\n\n"
               f"📊 تنبيهات اليوم: {d['alerts']}/{MAX_ALERTS_DAY}\n"
               f"📝 بعد الصفقة رد على الرسالة: ربحت / خسرت")
        
        mid_id = tg_send(txt)
        if mid_id is None:
            log.warning(f"فشل إرسال {sym}")
            continue
        
        lastmap[sym] = key
        d["alerts"] += 1
        st.setdefault("open", {})[str(mid_id)] = {
            "sym": sym, "dr": dr, "px": cl, "v4": v4,
            "zone": zone, "urgency": urgency
        }
        save_state(st)
        time.sleep(0.5)

# ─── Main ───
def main():
    st = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    old = st.get("day")
    if old and old.get("date") != today:
        send_day_summary(st, old)
    d = day_obj(st)
    if st.get("boot_date") != d["date"]:
        st["boot_date"] = d["date"]
        tg_send(f"🚀 بوت H2 بدأ (v4 Pro)\n"
                f"• أزواج: {len(SYMBOLS)}\n"
                f"• RSI Wilder + BB ddof=0 (مطابق TradingView)\n"
                f"• مناطق دخول + إلحاح تلقائي\n"
                f"• سقف تنبيهات: {MAX_ALERTS_DAY}/يوم\n"
                f"• سقف صفقات: {MAX_TRADES_DAY}/يوم\n"
                f"• توقف تلقائي: {STOP_AFTER_LOSSES} خسائر = {STOP_HOURS} ساعات")
    now = datetime.now(timezone.utc)
    wk = f"{now.isocalendar()[0]}-{now.isocalendar()[1]}"
    if now.weekday() == 0 and st.get("week_key") != wk:
        st["week_key"] = wk
        send_week_summary(st)
    listen(st)
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
            tg_send(f"🚨 *البوت انهار*\nالخطأ: `{e}`\nافحص Actions فوراً")
        except Exception:
            pass
        raise
