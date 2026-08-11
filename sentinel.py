import os
import json
import time
import datetime
import requests

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT", "")

TIMEFRAME = "30m"
RANGE = "7d"
TF_LABEL = "M30"
DURATION = "15 دقيقة"
TZ_OFFSET = 1
RSI_PERIOD = 14
PIVOT_LOOKBACK = 6
LEVEL_TOL = 0.0015
NEAR_TOL = 0.0012
MIN_SCORE = 3
COOLDOWN_SEC = 4 * 3600
MEM_FILE = "memory.json"

HOST = "https://query1.finance.yahoo.com"
PATH = "/v8/finance/chart/"

FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X",
    "EURCHF=X", "EURAUD=X", "EURCAD=X",
    "EURNZD=X", "GBPJPY=X", "GBPCHF=X",
    "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDCAD=X", "AUDCHF=X",
    "AUDNZD=X", "CADJPY=X", "CADCHF=X",
    "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
    "CHFJPY=X", "USDTRY=X", "USDMXN=X",
    "USDZAR=X", "USDSGD=X", "USDSEK=X",
    "USDNOK=X", "USDCNH=X",
]

mem = {}
if os.path.exists(MEM_FILE):
    try:
        mem = json.load(open(MEM_FILE))
    except Exception:
        mem = {}

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

def fetch_candles(symbol):
    url = HOST + PATH + symbol
    params = {
        "interval": TIMEFRAME,
        "range": RANGE,
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        url,
        params=params,
        headers=headers,
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

def sma_at(closes, p):
    if len(closes) < p:
        return None
    return sum(closes[-p:]) / p

def calc_rsi(closes, p):
    if len(closes) < p + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-p:]) / p
    al = sum(losses[-p:]) / p
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - (100 / (1 + rs))

def find_pivot_levels(candles, lb):
    sup = []
    res = []
    for i in range(lb, len(candles) - lb):
        w = candles[i-lb:i+lb+1]
        hi = max(c["h"] for c in w)
        lo = min(c["l"] for c in w)
        if candles[i]["h"] == hi:
            res.append(candles[i]["h"])
        if candles[i]["l"] == lo:
            sup.append(candles[i]["l"])
    return sup, res

def cluster_levels(levels, tol):
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lv in levels[1:]:
        last = clusters[-1][-1]
        if abs(lv - last) / lv <= tol:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    return [sum(c)/len(c) for c in clusters]

def scan_pair(symbol):
    name = symbol.replace("=X", "")
    candles = fetch_candles(symbol)
    if len(candles) < 120:
        return 0
    closed = candles[:-1]
    last_t = closed[-1]["t"]
    key = "last_" + name
    if mem.get(key) == last_t:
        return 0
    mem[key] = last_t

    closes = [x["c"] for x in closed]
    k = closed[-1]
    o = k["o"]
    h = k["h"]
    l = k["l"]
    c = k["c"]
    sma35 = sma_at(closes, 35)
    sma50 = sma_at(closes, 50)
    sma100 = sma_at(closes, 100)
    rsi = calc_rsi(closes, RSI_PERIOD)
    rsi_prev = calc_rsi(closes[:-1], RSI_PERIOD)
    if None in (sma35, sma50, sma100, rsi, rsi_prev):
        return 0

    sup, res = find_pivot_levels(
        closed[-200:], PIVOT_LOOKBACK
    )
    sups = cluster_levels(sup, LEVEL_TOL)
    ress = cluster_levels(res, LEVEL_TOL)

    rng = (h - l) or 1e-9
    body = c - o

    cs = 0
    cr = []
    if sma50 > sma100:
        cs += 1
        cr.append("ترند أعلى صاعد")
    if l <= sma35 <= c:
        cs += 1
        cr.append("ارتداد من SMA35")
    for s in sups[-4:]:
        if l <= s*(1+NEAR_TOL) and c > s:
            cs += 1
            cr.append("ارتداد من دعم")
            break
    if body > 0 and body >= 0.5*rng:
        cs += 1
        cr.append("شمعة صاعدة قوية")
    if rsi_prev < 30 <= rsi:
        cs += 1
        cr.append("خروج من تشبع بيعي")
    elif 45 <= rsi <= 65:
        cs += 1
        cr.append("RSI صحية للصعود")

    ps = 0
    pr = []
    if sma50 < sma100:
        ps += 1
        pr.append("ترند أعلى هابط")
    if h >= sma35 >= c:
        ps += 1
        pr.append("رفض من SMA35")
    for r in ress[-4:]:
        if h >= r*(1-NEAR_TOL) and c < r:
            ps += 1
            pr.append("رفض من مقاومة")
            break
    if body < 0 and -body >= 0.5*rng:
        ps += 1
        pr.append("شمعة هابطة قوية")
    if rsi_prev > 70 >= rsi:
        ps += 1
        pr.append("خروج من تشبع شرائي")
    elif 35 <= rsi <= 55:
        ps += 1
        pr.append("RSI صحية للهبوط")

    if cs == ps:
        return 0
    if cs > ps:
        side = "صعود 🟢 (CALL)"
        score = cs
        reasons = cr
    else:
        side = "هبوط 🔴 (PUT)"
        score = ps
        reasons = pr
    if score < MIN_SCORE:
        return 0
    cd_key = "cd_" + name
    now = time.time()
    if now - mem.get(cd_key, 0) < COOLDOWN_SEC:
        return 0
    mem[cd_key] = int(now)

    stars = "⭐" * score
    why = " + ".join(reasons)
    msg = (
        "📊 توصية تداول جديدة\n"
        "\n"
        "• الزوج: " + name + "\n"
        "• الفريم: " + TF_LABEL + "\n"
        "• مدة الصفقة: " + DURATION + "\n"
        "• الوقت: " + now_hhmm() + "\n"
        "• الاتجاه: " + side + "\n"
        "• القوة: " + stars
        + " (" + str(score) + "/5)\n"
        "• الأسباب: " + why + "\n"
        "\n"
        "💡 توصية آلية تعليمية"
        " — القرار النهائي لك"
    )
    send(msg)
    return 1

if __name__ == "__main__":
    ok = 0
    fail = 0
    sent = 0
    for pair in FOREX_PAIRS:
        try:
            sent += scan_pair(pair)
            ok += 1
        except Exception as e:
            fail += 1
            print("⚠️", pair, type(e).__name__, e)
        time.sleep(0.8)
    json.dump(mem, open(MEM_FILE, "w"))
    print("✅ فحص:", ok, "تمام /",
          fail, "تجاوز / 🎯 إشارات:", sent)
