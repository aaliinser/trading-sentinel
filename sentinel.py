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
        return
    try:
        a = "https://api.telegram.org/bot"
        j = {"chat_id": CH, "text": "🚨 " + msg}
        url = a + TG + "/sendMessage"
        requests.post(url, json=j, timeout=10)
    except Exception as e:
        print("TG ERR:", e)

def hhmm():
    t = datetime.datetime.utcnow()
    t += datetime.timedelta(hours=TZ)
    return t.strftime("%H:%M")

def market_open():
    t = datetime.datetime.utcnow()
    if t.weekday() >= 5:
        return False
    return True

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
        mem["day"] = cur
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
        m = "🧮 سقف 5 صفقات"
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
    if win:
        g = round(STAKE*PAY, 2)
        day["pnl"] = round(day["pnl"]+g, 2)
        day["loss"] = 0
        t = "✅ رابحة +" + str(g) + "$"
    else:
        day["pnl"] = round(day["pnl"]-STAKE, 2)
        day["loss"] += 1
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
            continue
        try:
            c5 = fetch(pr, "5m", "1d")
        except Exception:
            continue
        k = c5[-1]
        if sw.get("lt") == k["t"]:
            continue
        o = k["o"]
        h = k["h"]
        l = k["l"]
        c = k["c"]
        lvl = sw["lvl"]
        txt = "%.3f" % lvl if c > 50 else "%.5f" % lvl
        span = max(h - l, 1e-9)
        body_dn = o - c
        body_up = c - o
        lp = live_price(pr)
        lp_txt = ("%.3f" % lp if lp > 50 else "%.5f" % lp) if lp else "غير متوفر"
        if sw["dir"] == "PUT":
            touched = h >= lvl - 0.00025 * c
            rejected = body_dn >= 0.4 * span and c < lvl
            if touched and rejected:
                sw["lt"] = k["t"]
                mem["S_" + nm] = None
                send("⚡ ادخل الحين (سريع)!\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• المستوى: " + txt + "\n"
                     "• السعر الحي الآن: " + lp_txt + "\n"
                     "• الاتجاه: هبوط 🔴\n"
                     "• لمس + رفض على شمعة M5 جارية ✔️\n"
                     "• إذا السعر الحي قريب من المستوى → ادخل فورا\n"
                     "• مدة الصفقة: 15 دقيقة\n"
                     "• البروتوكول: غيث v6.14 LIVE")
            elif c > lvl + 0.0015 * c:
                mem["S_" + nm] = None
        else:
            touched = l <= lvl + 0.00025 * c
            rejected = body_up >= 0.4 * span and c > lvl
            if touched and rejected:
                sw["lt"] = k["t"]
                mem["S_" + nm] = None
                send("⚡ ادخل الحين (سريع)!\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• المستوى: " + txt + "\n"
                     "• السعر الحي الآن: " + lp_txt + "\n"
                     "• الاتجاه: صعود 🟢\n"
                     "• لمس + رفض على شمعة M5 جارية ✔️\n"
                     "• إذا السعر الحي قريب من المستوى → ادخل فورا\n"
                     "• مدة الصفقة: 15 دقيقة\n"
                     "• البروتوكول: غيث v6.14 LIVE")
            elif c < lvl - 0.0015 * c:
                mem["S_" + nm] = None
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
    o = k["o"]
    h = k["h"]
    l = k["l"]
    c = k["c"]
    rng = h-l
    if rng <= 0:
        return 0
    body = abs(c-o)
    e15 = ema(cl, 35)
    h1 = toh1(cd)
    e60 = ema([x["c"] for x in h1], 35)
    a15 = adx(cd)
    a60 = adx(h1)
    at = atrs(cd)
    if None in (e15, e60, a15, a60):
        return 0
    if not at:
        return 0
    atr = at[-1]
    aavg = sum(at[-20:])/min(20, len(at))
    rp, rn = rsi2(cl)
    if rn is None:
        return 0
    sup, res = pivots(cd)
    f = fun()
    f["ev"] += 1
    if a60 <= 15 or a15 <= 15:
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
    rej_c = True
    rej_p = True
    lo20 = min(x["l"] for x in cd[-20:])
    hi20 = max(x["h"] for x in cd[-20:])
    step = 0.5 if c > 50 else 0.005
    rb = (c//step)*step
    rt = rb + step
    near_s = sup is not None
    if near_s:
        near_s = abs(l-sup)/c*100 <= 0.5
    near_s = near_s or abs(l-rb)/c*100 <= 0.15
    near_r = res is not None
    if near_r:
        near_r = abs(h-res)/c*100 <= 0.5
    near_r = near_r or abs(h-rt)/c*100 <= 0.15
    side = None
    kind = ""
    c1 = up60 and c > e15 and near_s
    p1 = dn60 and c < e15 and near_r
    mid_c = c1 and rn <= 55 and rn > rp and c > o
    if mid_c:
        mid_c = (hi20-c) > 0.3*atr
    mid_p = p1 and rn >= 45 and rn < rp and c < o
    if mid_p:
        mid_p = (c-lo20) > 0.3*atr
    if mid_c or mid_p:
        f["f2"] += 1
    if mid_c and rej_c and not red3:
        side = "صعود 🟢 (CALL)"
        kind = "CALL"
    elif mid_p and rej_p and not grn3:
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
                send("👀 تنبيه تجهيز\n"
                     "\n"
                     "• الزوج: " + nm + "\n"
                     "• المستوى المستدير: " + wtxt + "\n"
                     "• الاتجاه المتوقع: " + wt + "\n"
                     "• الخطة: إذا لمس المستوى وتكوّنت"
                     " شمعة تأكيد بنفس الاتجاه → كن جاهزاً!")
    if side is None:
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
    send("📊 توصية تداول جديدة 🚀\n"
         "\n"
         "• الزوج: " + nm + "\n"
         "• الفريم: " + TF_LABEL + "\n"
         "• مدة الصفقة: " + DUR + "\n"
         "• الوقت: " + hhmm() + "\n"
         "• الاتجاه: " + side + "\n"
         "• البروتوكول: غيث v6.14 LIVE\n"
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
        send("🌅 غيث v6.14 LIVE صاحي 📡 (ويكند محمي)")
    seen = 0
    while time.time() < start + 200:
        try:
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
