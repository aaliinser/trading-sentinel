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
            cd[i]["h"]-
