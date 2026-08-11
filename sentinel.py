import os
import json
import time
import datetime
import requests

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT", "")

TF_LABEL = "M15"
DURATION = "15 دقيقة"
TZ_OFFSET = 1
STAKE = 6
PAYOUT = 0.9
TARGET = 10
MAX_TRADES = 5
MAX_LOSS = 3
TRADE_SEC = 15 * 60
MEM_FILE = "memory.json"

HOST = "https://query1.finance.yahoo.com"
PATH = "/v8/finance/chart/"

PAIRS = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X",
    "USDCNH=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "USDMXN=X",
    "USDSEK=X",
]

mem = {}
if os.path.exists(MEM_FILE):
    try:
        mem = json.load(open(MEM_FILE))
    except Exception:
        mem = {}

def save_mem():
    json.dump(mem, open(MEM_FILE, "w"))

def now_hhmm():
    t = datetime.datetime.utcnow()
    t += datetime.timedelta(hours=TZ_OFFSET)
    return t.strftime("%H:%M")

def send(msg):
    print("🚨", msg)
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        api = "https://api.telegram.org/bot"
        requests.post(
            api + TG_TOKEN + "/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": "🚨 " + msg,
            },
            timeout=10,
        )
    except Exception as e:
        print("TG ERR:", e)

def fetch(symbol, interval, rng):
    url = HOST + PATH + symbol
    r = requests.get(
        url,
        params={
            "interval": interval,
            "range": rng,
        },
        headers={
            "User-Agent": "Mozilla/5.0",
        },
        timeout=15,
    )
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
        out.append({
            "t": ts[i] * 1000,
            "o": o, "h": h,
            "l": l, "c": c,
        })
    return out

def ema_last(vals, p):
    if len(vals) < p:
        return None
    k = 2.0 / (p + 1)
    e = sum(vals[:p]) / p
    for v in vals[p:]:
        e = v * k + e * (1 - k)
    return e

def rsi_last2(closes, p=14):
    if len(closes) < p + 2:
        return None, None
    gs = []
    ls = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gs.append(max(d, 0))
        ls.append(max(-d, 0))
    ag = sum(gs[:p]) / p
    al = sum(ls[:p]) / p
    vals = []
    for i in range(p, len(gs)):
        ag = (ag * (p-1) + gs[i]) / p
        al = (al * (p-1) + ls[i]) / p
        if al == 0:
            vals.append(100.0)
        else:
            vals.append(100 - 100/(1 + ag/al))
    if len(vals) < 2:
        return None, None
    return vals[-2], vals[-1]

def atr_series(candles, p=14):
    if len(candles) < p + 2:
        return []
    trs = []
    for i in range(1, len(candles)):
        pc = candles[i-1]["c"]
        h = candles[i]["h"]
        l = candles[i]["l"]
        trs.append(max(
            h - l,
            abs(h - pc),
            abs(l - pc)))
   
