#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — بوت إشارات حي (v5.5 Cooldown)
الاستراتيجية دون أي تغيير: RSI(14) Wilder + BB(20,2.0) ddof=0 | 5m / 15m
v5.5: تبريد لكل زوج — لا إشارة جديدة لنفس الزوج قبل حسم السابقة
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
STOP_AFTER_LOSSES = 3
STOP_HOURS = 4
STATE_FILE = "state_h2.json"

USER_TZ_OFFSET_H = 3
WIN_START_H = 9
BLACKOUT_END_H = 3
MONTH_START = "2026-09-15"
BREAKEVEN_WR = 52.63

AR_DAYS = ["الاثنين","الثلاثاء","الأربعاء","الخميس",
           "الجمعة","السبت","الأحد"]
AR_MONTHS = ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
             "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("H2v55")

# ─── Safe accessors ───
def safe_list(st, key):
    """Return list; replace None/corrupt values with empty list."""
    val = st.get(key)
    if not isinstance(val, list):
        st[key] = []
        return st[key]
    return val

def safe_dict(st, key):
    """Return dict; replace None/corrupt values with empty dict."""
    val = st.get(key)
    if not isinstance(val, dict):
        st[key] = {}
        return st[key]
    return val

def ar_day(ds):
    try:
        d = datetime.strptime(ds, "%Y-%m-%d")
        return AR_DAYS[d.weekday()]
    except Exception:
        return ""

def ar_month(m):
    return AR_MONTHS[m - 1]

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

# ─── Flatten yfinance MultiIndex columns ───
def flatten_columns(df):
    """
    Normalise yfinance columns regardless of MultiIndex layout.
    Handles both: ('Open', 'USDJPY=X') and ('USDJPY=X', 'Open')
    """
    OHLC = ("Open", "High", "Low", "Close")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c) for c in df.columns]
        return df
    df = df.copy()
    price_level = None
    for lvl in range(df.columns.nlevels):
        values = {str(v) for v in df.columns.get_level_values(lvl)}
        if values & set(OHLC):
            price_level = lvl
            break
    if price_level is None:
        price_level = df.columns.nlevels - 1
    df.columns = [str(c) for c in df.columns.get_level_values(price_level)]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df

def fetch5m(sym):
    for a in range(3):
        try:
            df = yf.Ticker(sym).history(period="3d", interval="5m",
                                        auto_adjust=False,
                                        actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = flatten_columns(df)
            required = ["Open", "High", "Low", "Close"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"missing columns: {missing}")
            df = df[required]
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
    mid = c.rolling(BB_P, min_periods=BB_P).mean()
    sd = c.rolling(BB_P, min_periods=BB_P).std(ddof=0)
    bu = mid + BB_K * sd
    bl = mid - BB_K * sd
    return rsi, bu, bl, sd

def fmt_sym(s):
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def fmt_px(v):
    v = float(v)
    return f"{v:.3f}" if v > 50 else f"{v:.5f}"

def calc_urgency(cl, band, sdv, direction):
    if sdv <= 0:
        return "unknown", "unknown", "⚠️", "خطأ في حساب الانحراف"
    if direction == "CALL":
        if cl <= band * 1.0003:
            return "strong", "enter_now", "✅", "ادخل فوراً خلال 60 ثانية\n(السعر عند الباند — فرصة مثالية)"
        elif cl <= band * 1.0015:
            return "mid", "enter_now", "⚡", "ادخل خلال 60 ثانية\n(السعر قريب من الباند)"
        else:
            return "far", "skip", "🚫", "تخطّ — السعر ابتعد عن الباند"
    else:
        if cl >= band * 0.9997:
            return "strong", "enter_now", "✅", "ادخل فوراً خلال 60 ثانية\n(السعر عند الباند — فرصة مثالية)"
        elif cl >= band * 0.9985:
            return "mid", "enter_now", "⚡", "ادخل خلال 60 ثانية\n(السعر قريب من الباند)"
        else:
            return "far", "skip", "🚫", "تخطّ — السعر ابتعد عن الباند"

def resolve_pending(st):
    pend = safe_list(st, "pending")
    if not pend:
        return
    now = datetime.now(timezone.utc)
    done = []
    for p in pend:
        try:
            exp = datetime.fromisoformat(p["exp"])
        except Exception:
            done.append(p)
            continue
        if now < exp + timedelta(seconds=60):
            continue
        df = fetch5m(p["sym"])
        target = exp - timedelta(minutes=5)
        exit_px = None
        if df is not None and target in df.index:
            exit_px = float(df.loc[target, "Close"])
        if exit_px is None:
            if now > exp + timedelta(minutes=35):
                log.warning(f"void pending {p['sym']}")
                done.append(p)
            continue
        entry = float(p["px"])
        if exit_px == entry:
            log.info(f"tie void {p['sym']}")
            done.append(p)
            continue
        win = exit_px > entry if p["dr"] == "CALL" else exit_px < entry
        sent = datetime.fromisoformat(p["sent"])
        loc = sent + timedelta(hours=USER_TZ_OFFSET_H)
        sim_list = safe_list(st, "sim")
        sim_list.append({
            "dl": loc.strftime("%Y-%m-%d"),
            "hl": loc.hour,
            "sym": p["sym"], "dr": p["dr"],
            "v4": bool(p.get("v4", False)),
            "win": bool(win),
            "px": entry, "ex": exit_px,
        })
        if len(sim_list) > 2000:
            st["sim"] = sim_list[-2000:]
        log.info(f"auto-resolved {p['sym']} {p['dr']} win={win}")
        rid = p.get("mid")
        try:
            rid = int(rid) if rid is not None else None
        except Exception:
            rid = None
        txt = (f"{'✅' if win else '❌'} *نتيجة تلقائية*: "
               f"{'ربحت' if win else 'خسرت'}\n"
               f"• الدخول: {fmt_px(entry)}\n"
               f"• الخروج: {fmt_px(exit_px)}\n"
               f"• ضمن V4: {'✅' if p.get('v4') else '➖'}")
        tg_send(txt, reply_to=rid)
        done.append(p)
    for p in done:
        if p in pend:
            pend.remove(p)

def send_report(st):
    sim = safe_list(st, "sim")
    man = safe_list(st, "log")
    if not sim and not man:
        tg_send("📊 لا بيانات مسجلة بعد")
        return
    n = len(sim)
    w = sum(1 for x in sim if x["win"])
    wr = round(100*w/n, 1) if n else 0.0
    v4 = [x for x in sim if x["v4"]]
    v4w = sum(1 for x in v4 if x["win"])
    v4wr = round(100*v4w/len(v4), 1) if v4 else 0.0
    bo = n - len(v4)
    bow = w - v4w
    bowr = round(100*bow/bo, 1) if bo else 0.0
    mn = len(man)
    mw = sum(1 for x in man if x["win"])
    mwr = round(100*mw/mn, 1) if mn else 0.0
    txt = (f"📊 *تقرير الأداء*\n\n"
           f"*تلقائي (دخول بسعر التوصية):*\n"
           f"• إشارات: *{n}* | فوز *{wr}%*\n"
           f"• V4 ✅: {len(v4)} | {v4wr}%\n"
           f"• أساس ➖: {bo} | {bowr}%\n\n"
           f"*ردودك اليدوية:*\n"
           f"• صفقات: {mn} | فوز {mwr}%\n\n"
           f"📌 V4 تُعتمد إذا تفوقت بنقطتين+ بعد 30+ إشارة")
    tg_send(txt)

def send_day_summary(st, day_str):
    sim = [x for x in safe_list(st, "sim") if x["dl"] == day_str]
    win_sig = [x for x in sim if x["hl"] >= WIN_START_H]
    out_sig = [x for x in sim if x["hl"] < WIN_START_H]
    man = [x for x in safe_list(st, "log") if x.get("date") == day_str]
    n = len(win_sig)
    w = sum(1 for x in win_sig if x["win"])
    l = n - w
    wr = round(100*w/n, 1) if n else 0.0
    v4 = [x for x in win_sig if x["v4"]]
    v4w = sum(1 for x in v4 if x["win"])
    v4wr = round(100*v4w/len(v4), 1) if v4 else 0.0
    mn = len(man)
    mw = sum(1 for x in man if x["win"])
    mwr = round(100*mw/mn, 1) if mn else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    pnl = round(w * win_payout - l * stake, 2)
    txt = (f"🌙 *ملخص يوم {ar_day(day_str)} {day_str}*\n"
           f"(نافذة العد: {WIN_START_H}:00 – 24:00 بتوقيتك)\n\n"
           f"• إشارات النافذة: *{n}*\n"
           f"• تلقائي: فوز {w} / خسارة {l} | *{wr}%*\n"
           f"• منها V4: {len(v4)} | {v4wr}%\n"
           f"• صافي تقديري: {pnl:+.2f}$\n"
           f"• إشارات خارج النافذة: {len(out_sig)}\n"
           f"• ردودك اليدوية: {mn} | {mwr}%\n\n"
           f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def send_week_summary(st, days):
    sim = [x for x in safe_list(st, "sim") if x["dl"] in days]
    if not sim:
        return
    n = len(sim)
    w = sum(1 for x in sim if x["win"])
    l = n - w
    wr = 100*w/n
    p_star = 1.0/1.9
    se = (p_star*(1-p_star)/n) ** 0.5
    z = (wr/100 - p_star)/se if se > 0 else 0.0
    man = [x for x in safe_list(st, "log") if x.get("date") in days]
    mn = len(man)
    mw = sum(1 for x in man if x["win"])
    mwr = round(100*mw/mn, 1) if mn else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    txt = (f"📅 *ملخص أسبوع التداول*\n"
           f"({ar_day(days[0])} {days[0]} → "
           f"{ar_day(days[-1])} {days[-1]})\n\n"
           f"📆 *تفصيل الأيام:*\n")
    for dstr in days:
        ds = [x for x in sim if x["dl"] == dstr]
        dn = len(ds)
        dw = sum(1 for x in ds if x["win"])
        dwr = round(100*dw/dn, 1) if dn else 0.0
        txt += f"• {ar_day(dstr)} {dstr}: {dn} إشارة | {dwr}%\n"
    txt += f"\n🧩 *تفصيل الأزواج:*\n"
    for sym in sorted({x["sym"] for x in sim}):
        ss = [x for x in sim if x["sym"] == sym]
        sn = len(ss)
        sw = sum(1 for x in ss if x["win"])
        swr = round(100*sw/sn, 1) if sn else 0.0
        txt += f"• {fmt_sym(sym)}: {sn} | {swr}%\n"
    pnl = round(w * win_payout - l * stake, 2)
    txt += (f"\n• إجمالي الإشارات: *{n}*\n"
            f"• تلقائي: فوز {w} / خسارة {l} | *{round(wr,1)}%*\n"
            f"• صافي تقديري: {pnl:+.2f}$\n"
            f"• Z-score: {z:+.2f} "
            f"{'✅ دلالة حقيقية' if z > 1.96 else '⚠️ ضمن الضجيج'}\n"
            f"• ردودك اليدوية: {mn} | {mwr}%\n\n"
            f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

def send_month_summary(st, y, m):
    prefix = f"{y:04d}-{m:02d}"
    sim = [x for x in safe_list(st, "sim")
           if x["dl"].startswith(prefix) and x["dl"] >= MONTH_START]
    if not sim:
        tg_send(f"🗓️ *ملخص شهر {ar_month(m)} {y}*\n"
                f"لا إشارات مسجلة لهذا الشهر")
        return
    n = len(sim)
    w = sum(1 for x in sim if x["win"])
    l = n - w
    wr = 100*w/n
    p_star = 1.0/1.9
    se = (p_star*(1-p_star)/n) ** 0.5
    z = (wr/100 - p_star)/se if se > 0 else 0.0
    win_payout = 6.0 * 0.90
    stake = 6.0
    pnl = round(w * win_payout - l * stake, 2)
    days_sorted = sorted({x["dl"] for x in sim})
    txt = (f"🗓️ *ملخص شهر {ar_month(m)} {y}*\n"
           f"({days_sorted[0]} → {days_sorted[-1]})\n\n"
           f"📆 *تفصيل الأيام:*\n")
    for dstr in days_sorted:
        ds = [x for x in sim if x["dl"] == dstr]
        dn = len(ds)
        dw = sum(1 for x in ds if x["win"])
        dwr = round(100*dw/dn, 1) if dn else 0.0
        txt += f"• {ar_day(dstr)} {dstr}: {dn} | {dwr}%\n"
    txt += (f"\n• إجمالي الإشارات: *{n}*\n"
            f"• فوز {w} / خسارة {l} | *{round(wr,1)}%*\n"
            f"• صافي تقديري: {pnl:+.2f}$\n"
            f"• Z-score: {z:+.2f} "
            f"{'✅ دلالة حقيقية' if z > 1.96 else '⚠️ ضمن الضجيج'}\n\n"
            f"📌 التعادل: {BREAKEVEN_WR}%")
    tg_send(txt)

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
            op = safe_dict(st, "open").get(mid)
            d = day_obj(st)
            if op is None:
                tg_send("⚠️ رد على رسالة الإشارة مباشرة")
                continue
            del st["open"][mid]
            safe_list(st, "log").append({
                "date": d["date"],
                "sym": op.get("sym", "?"),
                "dr": op.get("dr", "?"),
                "v4": bool(op.get("v4", False)),
                "win": bool(win),
            })
            log_list = safe_list(st, "log")
            if len(log_list) > 600:
                st["log"] = log_list[-600:]
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
            win_payout = 6.0 * 0.90
            stake = 6.0
            pnl = round(d["wins"] * win_payout - d["losses"] * stake, 2)
            tg_send(f"💰 سُجلت: {'✅' if win else '❌'}\n"
                    f"• اليوم: {d['trades']} صفقة | فوز {wr}%\n"
                    f"• صافي تقديري: {pnl:+.2f}$")
        st["tg_offset"] = off
    except Exception as e:
        log.warning(f"listen: {e}")

def scan(st):
    d = day_obj(st)
    if time.time() < st.get("stop_until", 0):
        log.info("موقوف مؤقتاً (خسائر متتالية)")
        return
    lastmap = safe_dict(st, "last_sig")
    pend = safe_list(st, "pending")
    for sym in SYMBOLS:
        # ─── v5.5 Cooldown: no new signal while same symbol unresolved ───
        if any(p.get("sym") == sym for p in pend):
            log.info(f"{sym}: تبريد نشط — تخطي")
            continue
        df = fetch5m(sym)
        if df is None or len(df) < 60:
            continue
        rsi, bbu, bbl, sd = indicators(df)
        i = len(df) - 1
        ct = df.index[i]
        key = ct.isoformat()
        if lastmap.get(sym) == key:
            continue
        if np.isnan(sd.iloc[i]):
            continue
        entry_loc = ct + timedelta(minutes=5)
        entry_loc = entry_loc + timedelta(hours=USER_TZ_OFFSET_H)
        if entry_loc.weekday() == 0 and entry_loc.hour < BLACKOUT_END_H:
            log.info(f"{sym}: blackout افتتاح الأسبوع — تخطي")
            lastmap[sym] = key
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
        over = abs(cl - band_ref) / sdv if sdv > 0 else 0.0
        v4 = over >= 0.5
        if dr == "CALL":
            strong_entry = bl
            mid_entry = (cl + bl) / 2
        else:
            strong_entry = bu
            mid_entry = (cl + bu) / 2
        zone, urgency, emoji, urg_text = calc_urgency(cl, band_ref, sdv, dr)
        if urgency == "skip":
            log.info(f"{sym}: إشارة ضعيفة — تخطي")
            lastmap[sym] = key
            continue
        exp = ct + timedelta(minutes=5 + EXPIRY_MIN)
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
               f"📌 *السبب:* RSI {r:.1f} + السعر "
               f"{'تحت' if dr=='CALL' else 'فوق'} الباند بـ {over:.1f}σ\n"
               f"🏷️ ضمن V4: {'✅' if v4 else '➖'}\n\n"
               f"⌛ الانتهاء: 15 دقيقة (حتى {exp.strftime('%H:%M')} UTC)\n"
               f"💰 شرط الـ payout: 85% فأعلى فقط\n\n"
               f"📊 تنبيهات اليوم: {d['alerts']}\n"
               f"📝 النتيجة تُسجل تلقائياً عند الانتهاء")
        mid_id = tg_send(txt)
        if mid_id is None:
            log.warning(f"فشل إرسال {sym}")
            continue
        lastmap[sym] = key
        d["alerts"] += 1
        safe_dict(st, "open")[str(mid_id)] = {
            "sym": sym, "dr": dr, "px": cl, "v4": v4,
            "zone": zone, "urgency": urgency
        }
        safe_list(st, "pending").append({
            "mid": mid_id, "sym": sym, "dr": dr,
            "px": cl, "v4": v4,
            "exp": (ct + timedelta(minutes=5 + EXPIRY_MIN)).isoformat(),
            "sent": (ct + timedelta(minutes=5)).isoformat(),
        })
        save_state(st)
        time.sleep(0.5)

def main():
    st = load_state()
    loc = datetime.now(timezone.utc) + timedelta(hours=USER_TZ_OFFSET_H)
    today_local = loc.strftime("%Y-%m-%d")

    last_daily = st.get("last_daily")
    if last_daily is None:
        st["last_daily"] = today_local
    elif last_daily != today_local:
        prev = (loc - timedelta(days=1)).strftime("%Y-%m-%d")
        send_day_summary(st, prev)
        st["last_daily"] = today_local

    if st.get("last_month") is None:
        st["last_month"] = loc.strftime("%Y-%m")
    if loc.day == 1 and st.get("last_month") != loc.strftime("%Y-%m"):
        if loc.month > 1:
            py, pm = loc.year, loc.month - 1
        else:
            py, pm = loc.year - 1, 12
        send_month_summary(st, py, pm)
        st["last_month"] = loc.strftime("%Y-%m")

    now_utc = datetime.now(timezone.utc)
    iso = loc.isocalendar()
    wk = f"{iso[0]}-W{iso[1]}"
    if now_utc.weekday() == 5 and st.get("last_week") != wk:
        monday = loc - timedelta(days=loc.weekday())
        days = [(monday + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(5)]
        send_week_summary(st, days)
        st["last_week"] = wk

    d = day_obj(st)
    if st.get("boot_date") != d["date"]:
        st["boot_date"] = d["date"]
        tg_send(f"🚀 بوت H2 بدأ (v5.5 Cooldown)\n"
                f"• أزواج: {len(SYMBOLS)}\n"
                f"• بدون سقف تنبيهات — لن تضيع إشارة\n"
                f"• تبريد لكل زوج: إشارة واحدة حتى حسم النتيجة\n"
                f"• تسجيل تلقائي + رد تلقائي بالنتيجة\n"
                f"• ملخص يومي عند منتصف الليل\n"
                f"• ملخص أسبوعي السبت (بالأيام والتواريخ والأزواج)\n"
                f"• ملخص شهري أول كل شهر (العد من {MONTH_START})\n"
                f"• حظر إشارات الاثنين قبل {BLACKOUT_END_H:02d}:00\n"
                f"• الحماية: 3 خسائر متتالية = 4 ساعات")
    listen(st)
    resolve_pending(st)
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
