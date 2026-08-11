import os, json, time
import datetime, requests

TG = os.environ.get("TG_TOKEN", "")
CH = os.environ.get("TG_CHAT", "")
TF_LABEL = "M15"
DUR = "15 دقيقة"
TZ = 1
STAKE = 6
PAY = 0.9
TARGET = 10
MAXT = 5
MAXL = 3
TSEC = 900
MEMF = "memory.json"
HOST = "https://query1.finance.yahoo.com"
PATH = "/v8/finance/chart/"
PAIRS = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X",
    "USDCNH=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "USDMXN=X",
    "USDSEK=X",
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
        requests.post(
            a + TG + "/sendMessage",
            json={"chat_id": CH,
                  "text": "🚨 " + msg},
            timeout=10)
    except Exception as e:
        print("TG ERR:", e)

def hhmm():
    t = datetime.datetime.utcnow()
    t += datetime.timedelta(hours=TZ)
    return t.strftime("%H:%M")

def fetch(sym, itv, rng):
    r = requests.get(
        HOST + PATH + sym,
        params={"interval": itv,
                "range": rng},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15)
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
        out.append({"t": ts[i]*1000,
                    "o": o, "h": h,
                    "l": l, "c": c})
    return out

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
        tr.append(max(
            cd[i]["h"]-cd[i]["l"],
            abs(cd[i]["h"]-pc),
            abs(cd[i]["l"]-pc)))
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
        tr.append(max(
            h-l, abs(h-pc), abs(l-pc)))
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
    sup = None
    res = None
    top = len(cd)-1-lb
    for i in range(top, lb-1, -1):
        w = cd[i-lb:i+lb+1]
        hi = max(x["h"] for x in w)
        lo = min(x["l"] for x in w)
        if res is None and cd[i]["h"] == hi:
            res = hi
        if sup is None and cd[i]["l"] == lo:
            sup = lo
        if sup and res:
            break
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
        out.append({
            "o": g[0]["o"],
            "h": max(x["h"] for x in g),
            "l": min(x["l"] for x in g),
            "c": g[-1]["c"]})
    return out

def kz(t):
    dt = datetime.datetime.utcfromtimestamp(
        t/1000)
    h = dt.hour
    return 7 <= h < 10 or 12 <= h < 15

def getday():
    d = datetime.datetime.utcnow().strftime(
        "%Y-%m-%d")
    cur = mem.get("day", {})
    if cur.get("date") != d:
        cur = {"date": d, "trades": 0,
               "loss": 0, "pnl": 0.0,
               "stop": 0}
        mem["day"] = cur
    return cur

def daystop():
    day = getday()
    if day.get("stop"):
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
    if day.get("stop"):
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
        day["pnl"] = round(
            day["pnl"]-STAKE, 2)
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

def scan(sym):
    nm = sym.replace("=X", "")
    c15 = fetch(sym, "15m", "7d")
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
    lw = min(o, c)-l
    uw = h-max(o, c)
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
    ct = lt+900000
    if a60 <= 20 or a15 <= 20:
        return 0
    if atr <= 0.5*aavg:
        return 0
    if atr >= 2*aavg:
        return 0
    if abs(c-e15) > 1.5*atr:
        return 0
    if not kz(ct):
        return 0
    l3 = cd[-3:]
    red3 = all(
        x["c"] < x["o"] for x in l3)
    grn3 = all(
        x["c"] > x["o"] for x in l3)
    up60 = c > e60
    dn60 = c < e60
    rej_c = (lw >= 2*body
             and lw >= 0.6*rng
             and uw <= 0.15*rng)
    rej_p = (uw >= 2*body
             and uw >= 0.6*rng
             and lw <= 0.15*rng)
    near_s = sup is not None
    if near_s:
        near_s = abs(l-sup)/c*100 <= 0.2
    near_r = res is not None
    if near_r:
        near_r = abs(h-res)/c*100 <= 0.2
    brk_c = (c > o
             and body >= 0.7*rng
             and lw+uw <= 0.15*rng
             and o < e15 and c > e15
             and atr > 1.2*aavg
             and a15 > 25)
    brk_p = (c < o
             and body >= 0.7*rng
             and lw+uw <= 0.15*rng
             and o > e15 and c < e15
             and atr > 1.2*aavg
             and a15 > 25)
    side = None
    kind = ""
    if (up60 and c > e15 and near_s
            and rn <= 40 and rn > rp
            and rej_c and not red3):
        side = "صعود 🟢 (CALL)"
        kind = "CALL"
    elif (dn60 and c < e15 and near_r
            and rn >= 60 and rn < rp
            and rej_p and not grn3):
        side = "هبوط 🔴 (PUT)"
        kind = "PUT"
    elif brk_c and up60 and not red3:
        side = "اختراق صاعد 🚀 (BRK CALL)"
        kind = "BRK-CALL"
    elif brk_p and dn60 and not grn3:
        side = "اختراق هابط 💥 (BRK PUT)"
        kind = "BRK-PUT"
    if side is None:
        return 0
    if not cantrade():
        return 0
    day = getday()
    day["trades"] += 1
    mem["lock"] = time.time()+TSEC+60
    mem["pend"] = {
        "sym": sym, "nm": nm,
        "sd": kind, "en": c,
        "ev": time.time()+TSEC}
    send("📊 توصية تداول جديدة\n"
         "\n"
         "• الزوج: " + nm + "\n"
         "• الفريم: " + TF_LABEL + "\n"
         "• مدة الصفقة: " + DUR + "\n"
         "• الوقت: " + hhmm() + "\n"
         "• الاتجاه: " + side + "\n"
         "• البروتوكول: غيث v2.0 ✅\n"
         "• صفقة اليوم: "
         + str(day["trades"]) + "/5\n"
         "\n"
         "💡 توصية آلية تعليمية"
         " — القرار النهائي لك")
    daystop()
    return 1

if __name__ == "__main__":
    pending()
    ok = 0
    fl = 0
    st = 0
    for pr in PAIRS:
        try:
            st += scan(pr)
            ok += 1
        except Exception as e:
            fl += 1
            print("⚠️", pr,
                  type(e).__name__, e)
        time.sleep(0.8)
    json.dump(mem, open(MEMF, "w"))
    print("✅ غيث v2.0:", ok, "ok /",
          fl, "err / 🎯", st)
