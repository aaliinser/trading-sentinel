import os, json, time
import datetime, requests

TG = os.environ.get("TG_TOKEN", "")
CH = os.environ.get("TG_CHAT", "")
TF_LABEL = "M15"
DUR = "15 دقيقة"
TZ = 1
STAKE = 6
PAY = 0.9
TARGET = 999999
MAXT = 999999
MAXL = 999999
TSEC = 900
MAX_DEV = 0.0006
MAX_AHEAD = 0.0004
MEMF = "memory.json"
HOST = "https://query1.finance.yahoo.com"
PATH = "/v8/finance/chart/"
PAIRS = [
    "USDJPY=X", "AUDJPY=X", "EURJPY=X",
    "EURUSD=X", "GBPUSD=X", "EURGBP=X",
    "CADJPY=X", "EURCAD=X", "GBPCAD=X",
    "AUDCHF=X", "AUDUSD=X", "USDCHF=X",
    "CHFJPY=X", "AUDCAD=X", "USDCAD=X",
    "EURAUD=X", "EURCHF=X", "GBPJPY=X",
    "GBPCHF=X", "GBPAUD=X",
]
mem = {}
if os.path.exists(MEMF):
    try:
        mem = json.load(open(MEMF))
    except Exception:
        mem = {}

def send(msg):
    print("🚨", msg)
    if not TG or not CH:
        return None
    try:
        a = "https://api.telegram.org/bot"
        j = {"chat_id": CH, "text": "🚨 " + msg}
        url = a + TG + "/sendMessage"
        r = requests.post(url, json=j, timeout=10)
        try:
            return r.json().get("result", {}).get("message_id")
        except Exception:
            return None
    except Exception as e:
        print("TG ERR:", e)
        return None

def hhmm():
    t = datetime.datetime.utcnow()
    t += datetime.timedelta(hours=TZ)
    return t.strftime("%H:%M")

def market_open():
    t = datetime.datetime.utcnow()
    if t.weekday() >= 5:
        return False
    if t.weekday() == 4 and t.day <= 7 and 12 <= t.hour < 15:
        return False
    return 8 <= t.hour < 17

def fetch(sym, itv, rng):
    url = HOST + PATH + sym
    pp = {"interval": itv, "range": rng}
    hh = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, pp, headers=hh, timeout=15)
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    ts = d["timestamp"]
    q = d["indicators"]["quote"][0]
    out = []
    for i in range(len(ts)):
        o = q["open"][i]
        h = q["high"][i]
        l = q["low"][i]
        c = q["close"][i]
        if None in (o, h, l, c):
            continue
        dd = {"t": ts[i]*1000, "o": o}
        dd["h"] = h
        dd["l"] = l
        dd["c"] = c
        out.append(dd)
    return out

def live_price(sym):
    try:
        c1 = fetch(sym, "1m", "1d")
        return c1[-1]["c"]
    except Exception:
        return None

def ema(v, p):
    if len(v) < p:
        return None
    k = 2.0/(p+1)
    e = sum(v[:p])/p
    for x in v[p:]:
        e = x*k + e*(1-k)
    return e

def _emas(v, p):
    k = 2.0/(p+1)
    e = sum(v[:p])/p
    out = [e]
    for x in v[p:]:
        e = x*k + e*(1-k)
        out.append(e)
    return out

def macd2(cl):
    if len(cl) < 60:
        return None, None
    a = _emas(cl, 12)
    b = _emas(cl, 26)
    off = 26-12
    n = len(b)
    macd = [a[i+off]-b[i] for i in range(n)]
    if len(macd) < 12:
        return None, None
    sig = _emas(macd, 9)
    h1 = macd[len(macd)-2]-sig[len(sig)-2]
    h2 = macd[-1]-sig[-1]
    return h1, h2

def pat_call(k, p):
    r = k["h"]-k["l"]
    if r <= 0:
        return False
    lw = min(k["o"], k["c"])-k["l"]
    pin = lw >= 0.6*r
    eng = (k["c"] > k["o"]) and (p["c"] < p["o"])
    eng = eng and (k["c"] >= p["o"]) and (k["o"] <= p["c"])
    return pin or eng

def pat_put(k, p):
    r = k["h"]-k["l"]
    if r <= 0:
        return False
    uw = k["h"]-max(k["o"], k["c"])
    pin = uw >= 0.6*r
    eng = (k["c"] < k["o"]) and (p["c"] > p["o"])
    eng = eng and (k["c"] <= p["o"]) and (k["o"] >= p["c"])
    return pin or eng

def rsi2(cl, p=14):
    if len(cl) < p+2:
        return None, None
    g = []
    lo = []
    for i in range(1, len(cl)):
        d = cl[i] - cl[i-1]
        g.append(max(d, 0))
        lo.append(max(-d, 0))
    ag = sum(g[:p])/p
    al = sum(lo[:p])/p
    v = []
    for i in range(p, len(g)):
        ag = (ag*(p-1)+g[i])/p
        al = (al*(p-1)+lo[i])/p
        if al == 0:
            v.append(100.0)
        else:
            v.append(100-100/(1+ag/al))
    if len(v) < 2:
        return None, None
    return v[-2], v[-1]

def atrs(cd, p=14):
    if len(cd) < p+2:
        return []
    tr = []
    for i in range(1, len(cd)):
        pc = cd[i-1]["c"]
        h = cd[i]["h"]
        l = cd[i]["l"]
        m1 = h-l
        m2 = abs(h-pc)
        m3 = abs(l-pc)
        tr.append(max(m1, m2, m3))
    a = sum(tr[:p])/p
    out = [a]
    for x in tr[p:]:
        a = (a*(p-1)+x)/p
        out.append(a)
    return out

def adx(cd, p=14):
    n = len(cd)
    if n < p*3+1:
        return None
    tr = []
    ps = []
    ms = []
    for i in range(1, n):
        pc = cd[i-1]["c"]
        ph = cd[i-1]["h"]
        pl = cd[i-1]["l"]
        h = cd[i]["h"]
        l = cd[i]["l"]
        m1 = h-l
        m2 = abs(h-pc)
        m3 = abs(l-pc)
        tr.append(max(m1, m2, m3))
        up = h-ph
        dn = pl-l
        if up > dn and up > 0:
            ps.append(up)
        else:
            ps.append(0)
        if dn > up and dn > 0:
            ms.append(dn)
        else:
            ms.append(0)
    a = sum(tr[:p])
    sp = sum(ps[:p])
    sm = sum(ms[:p])
    dx = []
    for i in range(p, len(tr)):
        a = a-a/p+tr[i]
        sp = sp-sp/p+ps[i]
        sm = sm-sm/p+ms[i]
        pi = 100*sp/a if a else 0
        mi = 100*sm/a if a else 0
        s = pi+mi
        if s > 0:
            dx.append(100*abs(pi-mi)/s)
        else:
            dx.append(0)
    if len(dx) < p:
        return None
    x = sum(dx[:p])/p
    for d in dx[p:]:
        x = (x*(p-1)+d)/p
    return x

def pivots(cd, lb=5):
    sups = []
    ress = []
    top = len(cd)-1-lb
    for i in range(lb, top+1):
        w = cd[i-lb:i+lb+1]
        hi = max(x["h"] for x in w)
        lo = min(x["l"] for x in w)
        if cd[i]["h"] == hi:
            ress.append(hi)
        if cd[i]["l"] == lo:
            sups.append(lo)
    if not sups or not ress:
        return None, None
    last = cd[-1]
    price = last["c"]
    sup = None
    min_ds = 999.0
    for s in sups:
        ds = abs(s-price)/price*100
        if ds < min_ds and ds < 2.0:
            min_ds = ds
            sup = s
    res = None
    min_dr = 999.0
    for r in ress:
        dr = abs(r-price)/price*100
        if dr < min_dr and dr < 2.0:
            min_dr = dr
            res = r
    return sup, res

def toh1(c15):
    gr = {}
    for k in c15:
        hr = k["t"]//3600000
        gr.setdefault(hr, []).append(k)
    out = []
    for h in sorted(gr):
        g = gr[h]
        if len(g) < 4:
            continue
        dd = {"o": g[0]["o"]}
        dd["h"] = max(x["h"] for x in g)
        dd["l"] = min(x["l"] for x in g)
        dd["c"] = g[-1]["c"]
        out.append(dd)
    return out

def getday():
    d = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    cur = mem.get("day", {})
    if cur.get("date") != d:
        cur = {"date": d, "trades": 0}
        cur["loss"] = 0
        cur["pnl"] = 0.0
        cur["stop"] = 0
        cur["win"] = 0
        cur["lose"] = 0
        cur["sigs"] = 0
        cur["mwin"] = 0
        cur["mlose"] = 0
        mem["day"] = cur
    return cur

def getmonth():
    ym = datetime.datetime.utcnow().strftime("%Y-%m")
    cur = mem.get("month") or {}
    if cur.get("ym") != ym:
        cur = {"ym": ym, "win": 0, "lose": 0}
        cur["mwin"] = 0
        cur["mlose"] = 0
        cur["pnl"] = 0.0
        mem["month"] = cur
    return cur

def fun():
    d = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    cur = mem.get("fun", {})
    if cur.get("date") != d:
        cur = {"date": d, "ev": 0, "f1": 0}
        cur["f2"] = 0
        cur["f3"] = 0
        mem["fun"] = cur
    return cur

def daystop():
    day = getday()
    if False:
        return
    m = None
    if day["pnl"] >= TARGET:
        m = "🎯 هدف اليوم تحقق"
    elif day["loss"] >= MAXL:
        m = "🛑 3 خسائر متتالية"
    elif day["trades"] >= MAXT:
        m = "🧮 سقف الصفقات"
    if m:
        day["stop"] = 1
        send("⏸️ آلة إدارة اليوم\n"
             "\n"
             "• " + m + "\n"
             "• توقف بقية اليوم\n"
             "• صافي اليوم: "
             + str(day["pnl"]) + "$")

def cantrade():
    day = getday()
    if False:
        return False
    if day["trades"] >= MAXT:
        return False
    if day["loss"] >= MAXL:
        return False
    if day["pnl"] >= TARGET:
        return False
    if time.time() < mem.get("lock", 0):
        return False
    return True

def prune_trades():
    op = mem.get("open_trades", {})
    now = time.time()
    kill = []
    for mid in op:
        rc = op[mid]
        if rc.get("done") or now - rc.get("t", now) > 86400:
            kill.append(mid)
    for mid in kill:
        del op[mid]

def listen_replies():
    if not TG:
        return
    try:
        prune_trades()
        off = mem.get("upd_off", 0)
        url = "https://api.telegram.org/bot" + TG + "/getUpdates"
        pp = {"offset": off, "timeout": 0}
        r = requests.get(url, params=pp, timeout=10)
        data = r.json().get("result", [])
        day = getday()
        mo = getmonth()
        for u in data:
            nid = u.get("update_id", 0)
            if nid >= off:
                off = nid + 1
            msg = u.get("message") or u.get("edited_message")
            if not msg:
                continue
            txt_raw = msg.get("text") or ""
            txt = txt_raw.lower()
            win = None
            if any(w in txt for w in ["ربحت", "رابحة", "won", "win"]):
                win = True
            elif any(w in txt for w in ["خسرت", "خاسرة", "lost", "lose"]):
                win = False
            if win is None:
                continue
            op = mem.setdefault("open_trades", {})
            rec = None
            rep = msg.get("reply_to_message")
            if rep is not None:
                rc2 = op.get(str(rep.get("message_id")))
                if rc2 is not None and not rc2.get("done"):
                    rec = rc2
            if rec is None:
                for mid2 in op:
                    rc2 = op[mid2]
                    if rc2.get("done"):
                        continue
                    if rc2.get("nm", "") in txt_raw:
                        rec = rc2
                        break
            # fallback: استخدام آخر إشارة مرسلة خلال 30 دقيقة
            if rec is None and mem.get("last_sig"):
                ls = mem["last_sig"]
                if time.time() - ls.get("t", 0) < 1800:
                    for mid2 in op:
                        rc2 = op[mid2]
                        if rc2.get("done"):
                            continue
                        if rc2.get("nm") == ls.get("nm"):
                            rec = rc2
                            break
            if rec is None:
                send("⚠️ ما قدرت أربط ردك بصفقة مفتوحة\n"
                     "\n"
                     "• استخدم خاصية الرد (Reply) على رسالة الإشارة\n"
                     "• أو اكتب اسم الزوج مع النتيجة، مثال: EUR/USD ربحت")
                continue
            rec["done"] = True
            if win:
                day["mwin"] = day.get("mwin", 0)+1
                mo["mwin"] = mo.get("mwin", 0)+1
                g = round(STAKE*PAY, 2)
                day["pnl"] = round(day.get("pnl", 0.0)+g, 2)
                mo["pnl"] = round(mo.get("pnl", 0.0)+g, 2)
                t = "✅ رابحة +" + str(g) + "$"
            else:
                day["mlose"] = day.get("mlose", 0)+1
                mo["mlose"] = mo.get("mlose", 0)+1
                day["pnl"] = round(day.get("pnl", 0.0)-STAKE, 2)
                mo["pnl"] = round(mo.get("pnl", 0.0)-STAKE, 2)
                t = "❌ خاسرة -" + str(STAKE) + "$"
            send("💰 تم تسجيل صفقتك\n"
                 "\n"
                 "• الزوج: " + rec.get("nm", "?") + "\n"
                 "• النتيجة: " + t + "\n"
                 "• صافي اليوم: " + str(day["pnl"]) + "$\n"
                 "• صافي الشهر: " + str(mo["pnl"]) + "$")
        mem["upd_off"] = off
    except Exception as e:
        print("LISTEN ERR:", e)

def reports():
    now = datetime.datetime.utcnow()
    ym = now.strftime("%Y-%m")
    now2 = now + datetime.timedelta(hours=TZ)
    today = now2.strftime("%Y-%m-%d")
    hr = now2.hour
    mc = mem.get("month")
    if mc is not None and mc.get("ym") and mc["ym"] != ym:
        tw = mc.get("win", 0) + mc.get("mwin", 0)
        tl = mc.get("lose", 0) + mc.get("mlose", 0)
        tt = tw + tl
        rate = round(100*tw/tt) if tt > 0 else 0
        send("🗓️ جرد الشهر الكامل\n"
             "\n"
             "• الشهر: " + mc["ym"] + "\n"
             "• الآلي: " + str(mc.get("win", 0)) + "✅ / " + str(mc.get("lose", 0)) + "❌\n"
             "• اليدوي: " + str(mc.get("mwin", 0)) + "✅ / " + str(mc.get("mlose", 0)) + "❌\n"
             "• إجمالي الصفقات: " + str(tt) + "\n"
             "• معدل الفوز: " + str(rate) + "%\n"
             "• صافي الشهر: " + str(mc.get("pnl", 0.0)) + "$")
        mem["month"] = None
    mo = getmonth()
    m_line = "• صافي الشهر: " + str(mo.get("pnl", 0.0)) + "$"
    rd = mem.get("repdate")
    if rd is None:
        mem["repdate"] = today
    elif rd != today:
        d = mem.get("day", {})
        aw = d.get("win", 0)
        al = d.get("lose", 0)
        mw = d.get("mwin", 0)
        ml = d.get("mlose", 0)
        ar = round(100*aw/(aw+al)) if aw+al > 0 else 0
        mr = round(100*mw/(mw+ml)) if mw+ml > 0 else 0
        send("📊 جرد اليوم الكامل (24 ساعة)\n"
             "\n"
             "• التاريخ: " + rd + "\n"
             "• الآلي: " + str(aw) + "✅ / " + str(al) + "❌ (" + str(ar) + "%)\n"
             "• اليدوي: " + str(mw) + "✅ / " + str(ml) + "❌ (" + str(mr) + "%)\n"
             "• صافي اليوم: " + str(d.get("pnl", 0.0)) + "$\n"
             + m_line)
        mem["repdate"] = today
    if hr in (4, 8, 12, 16, 20):
        slot = today + "-" + str(hr)
        if mem.get("rep4") != slot:
            mem["rep4"] = slot
            d = mem.get("day", {})
            aw = d.get("win", 0)
            al = d.get("lose", 0)
            mw = d.get("mwin", 0)
            ml = d.get("mlose", 0)
            ar = round(100*aw/(aw+al)) if aw+al > 0 else 0
            mr = round(100*mw/(mw+ml)) if mw+ml > 0 else 0
            send("⏱️ جرد كل 4 ساعات\n"
                 "\n"
                 "• الآلي: " + str(aw) + "✅ / " + str(al) + "❌ (" + str(ar) + "%)\n"
                 "• اليدوي: " + str(mw) + "✅ / " + str(ml) + "❌ (" + str(mr) + "%)\n"
                 "• صافي اليوم: " + str(d.get("pnl", 0.0)) + "$\n"
                 + m_line)

def pending():
    p = mem.get("pend")
    if not p:
        return
    if time.time() < p["ev"]:
        return
    try:
        cx = fetch(p["sym"], "15m", "1d")
        pr = cx[-1]["c"]
    except Exception:
        p["ev"] = time.time()+120
        return
    if "CALL" in p["sd"]:
        win = pr > p["en"]
    else:
        win = pr < p["en"]
    day = getday()
    mo = getmonth()
    if win:
        g = round(STAKE*PAY, 2)
        day["pnl"] = round(day["pnl"]+g, 2)
        mo["pnl"] = round(mo.get("pnl", 0.0)+g, 2)
        day["loss"] = 0
        day["win"] = day.get("win", 0)+1
        mo["win"] = mo.get("win", 0)+1
        t = "✅ رابحة +" + str(g) + "$"
    else:
        day["pnl"] = round(day["pnl"]-STAKE, 2)
        mo["pnl"] = round(mo.get("pnl", 0.0)-STAKE, 2)
        day["loss"] += 1
        day["lose"] = day.get("lose", 0)+1
        mo["lose"] = mo.get("lose", 0)+1
        t = "❌ خاسرة -" + str(STAKE) + "$"
    send("💰 نتيجة الصفقة\n"
         "\n"
         "• الزوج: " + p["nm"] + "\n"
         "• الاتجاه: " + p["sd"] + "\n"
         "• النتيجة: " + t + "\n"
         "• صافي اليوم: "
         + str(day["pnl"]) + "$")
    mem["pend"] = None
    daystop()

# =====================================================================
# الدالة المُرقّعة: sniper v2 FINAL
# =====================================================================
def sniper():
    if not market_open():
        return
    for pr in PAIRS:
        nm = pr[0:3] + "/" + pr[3:6]
        sw = mem.get("S_" + nm)
        if sw is None:
            continue
        if time.time() - sw["t"] > 10800:
            mem["S_" + nm] = None
            send("⌛ انتهاء مراقبة\n"
                 "\n"
                 "• الزوج: " + nm + "\n"
                 "• مرت 3 ساعات بدون إشارة أو انقلاب\n"
                 "• أُغلقت المراقبة بهدوء ⌛")
            continue
        try:
            c5 = fetch(pr, "5m", "1d")
        except Exception:
            continue
        if len(c5) < 3:
            continue
        k = c5[-2]          # الشمعة المغلقة (وليس الوليدة)
        prev5 = c5[-3]
        if sw.get("lt") == k["t"]:
            continue
        sw["lt"] = k["t"]
        o = k["o"]
        h = k["h"]
        l = k["l"]
        c = k["c"]
        lvl = sw["lvl"]
        txt = "%.3f" % lvl if c > 50 else "%.5f" % lvl
        span = max(h - l, 1e-9)
        body_dn = o - c
        body_up = c - o
        pd = mem.get("pend")
        if pd is not None and pd.get("nm") == nm:
            continue
        try:
            c15x = fetch(pr, "15m", "1d")
            clx = [x["c"] for x in c15x[:-1]]
            rpx, rnx = rsi2(clx)
        except Exception:
            rnx = None
        lp = live_price(pr)
        lpv = lp if lp is not None else c
        lp_txt = ("%.3f" % lp if lp > 50 else "%.5f" % lp) if lp else "تقريبي (شمعة)"
        dev_dn = (lvl - lpv) / c
        dev_up = (lpv - lvl) / c

        # ========== PUT ==========
        if sw["dir"] == "PUT":
            touched = h >= lvl - 0.00025 * c
            rej = body_dn >= 0.4 * span or pat_put(k, prev5)
            rejected = rej and c < lvl
            rsi_ok = rnx is None or rnx >= 42
            flip = (c > lvl) and (body_up >= 0.5 * span)

            if flip:
                sw["dir"] = "CALL"
                sw["t"] = time.time()
                sw["qlog"] = 0
                sw["plog"] = 0
                send("🔄 انقلاب المستوى\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• المستوى: " + txt + "\n"
                     "• شمعة مغلقة قوية اخترقت المستوى صعوداً 🟢\n"
                     "• المستوى صار دعماً محتملاً\n"
                     "• الخطة الجديدة: ريتست + تأكيد صاعد → فضية CALL 🔄")
            elif touched and rejected and rsi_ok:
                if dev_dn > MAX_DEV:
                    if not sw.get("plog"):
                        sw["plog"] = 1
                        send("🛡️ حماية الانحراف\n"
                             "\n"
                             "• الزوج: " + nm + "\n"
                             "• المستوى: " + txt + "\n"
                             "• السعر الحي: " + lp_txt + "\n"
                             "• السعر بعيد الآن → لا مطاردة\n"
                             "• المراقبة باقية بانتظار الريتست 🛡️")
                elif dev_up > MAX_AHEAD:
                    if not sw.get("plog"):
                        sw["plog"] = 1
                        send("🛡️ حماية التبكير\n"
                             "\n"
                             "• الزوج: " + nm + "\n"
                             "• المستوى: " + txt + "\n"
                             "• السعر الحي: " + lp_txt + "\n"
                             "• السعر لسه ما وصل للمستوى بعد\n"
                             "• المراقبة باقية بانتظار لمس حقيقي 🛡️")
                else:
                    # قبل إطلاق الفضية: إغلاق سجل التجهيز القديم
                    op = mem.get("open_trades", {})
                    for mid2 in list(op.keys()):
                        rc2 = op[mid2]
                        if rc2.get("nm") == nm and not rc2.get("done") and rc2.get("type") == "prep":
                            rc2["done"] = True
                    mem["S_" + nm] = None
                    dd = getday()
                    dd["sigs"] = dd.get("sigs", 0)+1
                    mid = send("🥈 ادخل الحين (فضية)!\n"
                         "\n"
                         "• الزوج: " + nm + "\n"
                         "• المستوى: " + txt + "\n"
                         "• السعر الحي الآن: " + lp_txt + "\n"
                         "• الاتجاه: هبوط 🔴\n"
                         "• لمس + رفض على شمعة 5م مغلقة ✔️\n"
                         "• السعر داخل المنطقة الذهبية → ادخل فورا\n"
                         "• مدة الصفقة: 15 دقيقة\n"
                         "• البروتوكول: غيث v6.19 FULL\n"
                         "\n"
                         "📝 بعد الصفقة رد بـ: ربحت / خسرت")
                    if mid:
                        op = mem.setdefault("open_trades", {})
                        op[str(mid)] = {"nm": nm, "done": False, "t": time.time(), "type": "sig"}
                        mem["last_sig"] = {"nm": nm, "mid": mid, "t": time.time()}
            elif c > lvl + 0.0015 * c:
                mem["S_" + nm] = None
                send("🔇 إغلاق مراقبة\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• السعر ابتعد فوق المستوى بدون شمعة انقلاب قوية\n"
                     "• أُغلقت المراقبة 🔇")
            elif touched:
                if not sw.get("qlog"):
                    sw["qlog"] = 1
                    why = []
                    if not rejected:
                        why.append("لا رفض واضح (جسم/نمط)")
                    if not rsi_ok:
                        why.append("بوابة RSI (<42)")
                    send("🔍 لمس بدون إشارة\n"
                         "\n"
                         "• الزوج: " + nm + "\n"
                         "• المستوى: " + txt + "\n"
                         "• السبب: " + " + ".join(why) + "\n"
                         "• المراقبة مستمرة 🔍")

        # ========== CALL ==========
        else:
            touched = l <= lvl + 0.00025 * c
            rej = body_up >= 0.4 * span or pat_call(k, prev5)
            rejected = rej and c > lvl
            rsi_ok = rnx is None or rnx <= 58
            flip = (c < lvl) and (body_dn >= 0.5 * span)

            if flip:
                sw["dir"] = "PUT"
                sw["t"] = time.time()
                sw["qlog"] = 0
                sw["plog"] = 0
                send("🔄 انقلاب المستوى\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• المستوى: " + txt + "\n"
                     "• شمعة مغلقة قوية اخترقت المستوى هبوطاً 🔴\n"
                     "• المستوى صار مقاومة محتملة\n"
                     "• الخطة الجديدة: ريتست + تأكيد هابط → فضية PUT 🔄")
            elif touched and rejected and rsi_ok:
                if dev_up > MAX_DEV:
                    if not sw.get("plog"):
                        sw["plog"] = 1
                        send("🛡️ حماية الانحراف\n"
                             "\n"
                             "• الزوج: " + nm + "\n"
                             "• المستوى: " + txt + "\n"
                             "• السعر الحي: " + lp_txt + "\n"
                             "• السعر بعيد الآن → لا مطاردة\n"
                             "• المراقبة باقية بانتظار الريتست 🛡️")
                elif dev_dn > MAX_AHEAD:
                    if not sw.get("plog"):
                        sw["plog"] = 1
                        send("🛡️ حماية التبكير\n"
                             "\n"
                             "• الزوج: " + nm + "\n"
                             "• المستوى: " + txt + "\n"
                             "• السعر الحي: " + lp_txt + "\n"
                             "• السعر لسه ما وصل للمستوى بعد\n"
                             "• المراقبة باقية بانتظار لمس حقيقي 🛡️")
                else:
                    op = mem.get("open_trades", {})
                    for mid2 in list(op.keys()):
                        rc2 = op[mid2]
                        if rc2.get("nm") == nm and not rc2.get("done") and rc2.get("type") == "prep":
                            rc2["done"] = True
                    mem["S_" + nm] = None
                    dd = getday()
                    dd["sigs"] = dd.get("sigs", 0)+1
                    mid = send("🥈 ادخل الحين (فضية)!\n"
                         "\n"
                         "• الزوج: " + nm + "\n"
                         "• المستوى: " + txt + "\n"
                         "• السعر الحي الآن: " + lp_txt + "\n"
                         "• الاتجاه: صعود 🟢\n"
                         "• لمس + رفض على شمعة 5م مغلقة ✔️\n"
                         "• السعر داخل المنطقة الذهبية → ادخل فورا\n"
                         "• مدة الصفقة: 15 دقيقة\n"
                         "• البروتوكول: غيث v6.19 FULL\n"
                         "\n"
                         "📝 بعد الصفقة رد بـ: ربحت / خسرت")
                    if mid:
                        op = mem.setdefault("open_trades", {})
                        op[str(mid)] = {"nm": nm, "done": False, "t": time.time(), "type": "sig"}
                        mem["last_sig"] = {"nm": nm, "mid": mid, "t": time.time()}
            elif c < lvl - 0.0015 * c:
                mem["S_" + nm] = None
                send("🔇 إغلاق مراقبة\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• السعر ابتعد تحت المستوى بدون شمعة انقلاب قوية\n"
                     "• أُغلقت المراقبة 🔇")
            elif touched:
                if not sw.get("qlog"):
                    sw["qlog"] = 1
                    why = []
                    if not rejected:
                        why.append("لا رفض واضح (جسم/نمط)")
                    if not rsi_ok:
                        why.append("بوابة RSI (>58)")
                    send("🔍 لمس بدون إشارة\n"
                         "\n"
                         "• الزوج: " + nm + "\n"
                         "• المستوى: " + txt + "\n"
                         "• السبب: " + " + ".join(why) + "\n"
                         "• المراقبة مستمرة 🔍")
        time.sleep(0.3)

def scan(sym):
    if not market_open():
        return 0
    nm = sym[0:3] + "/" + sym[3:6]
    c15 = fetch(sym, "15m", "3d")
    if len(c15) < 150:
        return 0
    cd = c15[:-1]
    lt = cd[-1]["t"]
    if mem.get("L_"+nm) == lt:
        return 0
    mem["L_"+nm] = lt
    cl = [x["c"] for x in cd]
    k = cd[-1]
    pvc = cd[-2]
    o = k["o"]
    h = k["h"]
    l = k["l"]
    c = k["c"]
    rng = h-l
    if rng <= 0:
        return 0
    e15 = ema(cl, 35)
    h1 = toh1(cd)
    e60 = ema([x["c"] for x in h1], 35)
    e50 = ema([x["c"] for x in h1], 50)
    a15 = adx(cd)
    a60 = adx(h1)
    at = atrs(cd)
    mh1, mh2 = macd2(cl)
    if None in (e15, e60, e50, a15, a60):
        return 0
    if not at:
        return 0
    if mh1 is None:
        return 0
    atr = at[-1]
    aavg = sum(at[-20:])/min(20, len(at))
    rp, rn = rsi2(cl)
    if rn is None:
        return 0
    sup, res = pivots(cd)
    f = fun()
    f["ev"] += 1
    if a60 <= 22 or a15 <= 20:
        return 0
    if atr <= 0.4*aavg:
        return 0
    if atr >= 2.5*aavg:
        return 0
    if abs(c-e15) > 2.0*atr:
        return 0
    f["f1"] += 1
    l3 = cd[-3:]
    red3 = all(x["c"] < x["o"] for x in l3)
    grn3 = all(x["c"] > x["o"] for x in l3)
    up60 = c > e60
    dn60 = c < e60
    lo20 = min(x["l"] for x in cd[-20:])
    hi20 = max(x["h"] for x in cd[-20:])
    step = 0.5 if c > 50 else 0.005
    rb = (c//step)*step
    rt = rb + step
    near_s = sup is not None
    if near_s:
        near_s = abs(l-sup) <= 0.5*atr
    near_s = near_s or abs(l-rb)/c*100 <= 0.15
    near_r = res is not None
    if near_r:
        near_r = abs(h-res) <= 0.5*atr
    near_r = near_r or abs(h-rt)/c*100 <= 0.15
    side = None
    kind = ""
    c1 = up60 and c > e15 and near_s and c > e50
    p1 = dn60 and c < e15 and near_r and c < e50
    mid_c = c1 and 40 <= rn <= 55 and rn > rp
    if mid_c:
        mid_c = pat_call(k, pvc) and mh1 > 0 and mh2 > 0
    if mid_c:
        mid_c = (hi20-c) > 0.3*atr
    mid_p = p1 and 45 <= rn <= 60 and rn < rp
    if mid_p:
        mid_p = pat_put(k, pvc) and mh1 < 0 and mh2 < 0
    if mid_p:
        mid_p = (c-lo20) > 0.3*atr
    if mid_c or mid_p:
        f["f2"] += 1
    if mid_c and not red3:
        side = "صعود 🟢 (CALL)"
        kind = "CALL"
    elif mid_p and not grn3:
        side = "هبوط 🔴 (PUT)"
        kind = "PUT"
    if side is not None:
        f["f3"] += 1
    if side is None:
        wl = None
        wt = ""
        if up60 and abs(l-rb)/c*100 <= 0.025 and rn <= 60:
            wl = rb
            wt = "صعود 🟢"
        if wl is None and dn60 and abs(h-rt)/c*100 <= 0.025 and rn >= 40:
            wl = rt
            wt = "هبوط 🔴"
        if wl is not None:
            wtxt = "%.3f" % wl if c > 50 else "%.5f" % wl
            wk = "W_" + nm
            now = time.time()
            lw = mem.get(wk)
            okw = True
            if lw is not None:
                if lw[0] == wtxt and now - lw[1] < 3600:
                    okw = False
            if okw:
                mem[wk] = [wtxt, now]
                mem["S_" + nm] = {"lvl": wl, "t": now}
                mem["S_" + nm]["dir"] = "CALL" if "صعود" in wt else "PUT"
                mid = send("👀 تنبيه تجهيز\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• المستوى المستدير: " + wtxt + "\n"
                     "• الاتجاه المتوقع: " + wt + "\n"
                     "• الخطة: إذا لمس المستوى وتكوّنت"
                     " شمعة تأكيد بنفس الاتجاه → كن جاهزاً!")
                if mid:
                    op = mem.setdefault("open_trades", {})
                    op[str(mid)] = {"nm": nm, "done": False, "t": time.time(), "type": "prep"}
                    mem["last_sig"] = {"nm": nm, "mid": mid, "t": time.time()}
    if side is None:
        return 0
    if mem.get("S_" + nm) is not None:
        return 0
    if not cantrade():
        return 0
    day = getday()
    day["trades"] += 1
    mem["lock"] = time.time()+TSEC+60
    mem["pend"] = {"sym": sym, "nm": nm}
    mem["pend"]["sd"] = kind
    mem["pend"]["en"] = c
    mem["pend"]["ev"] = time.time()+TSEC
    send("🟢 توصية ذهبية 🚀\n"
         "\n"
         "• الزوج: " + nm + "\n"
         "• الفريم: " + TF_LABEL + "\n"
         "• مدة الصفقة: " + DUR + "\n"
         "• الوقت: " + hhmm() + "\n"
         "• الاتجاه: " + side + "\n"
         "• تأكيد: 3 فريمات + MACD + نمط شمعة ✔️\n"
         "• البروتوكول: غيث v6.19 FULL\n"
         "• صفقة اليوم: "
         + str(day["trades"]) + "\n"
         "\n"
         "💡 توصية آلية تعليمية")
    daystop()
    return 1

if __name__ == "__main__":
    start = time.time()
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    if mem.get("boot") != today:
        mem["boot"] = today
        send("🌅 غيث v6.19 FULL صاحي 🛡️ (درع باتجاهين)")
    seen = 0
    while time.time() < start + 200:
        try:
            reports()
            listen_replies()
            pending()
            sniper()
            rc = fetch("EURUSD=X", "15m", "1d")
            ct = rc[-2]["t"]
        except Exception:
            ct = seen
        if ct != seen:
            seen = ct
            ok = 0
            fl = 0
            st = 0
            for pr in PAIRS:
                try:
                    st += scan(pr)
                    ok += 1
                except Exception as e:
                    fl += 1
                    print("⚠️", pr, type(e).__name__, e)
                time.sleep(0.2)
            json.dump(mem, open(MEMF, "w"))
            print("✅ دورة:", ok, "ok /", fl, "err / 🎯", st)
        json.dump(mem, open(MEMF, "w"))
        time.sleep(60)
