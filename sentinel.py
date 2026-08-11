import os, json, time, requests

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT  = os.environ.get("TG_CHAT", "")
SYMBOL   = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL = os.environ.get("INTERVAL", "1m")
SMA_PERIOD, RSI_PERIOD, PIVOT_LOOKBACK = 35, 14, 5
LEVEL_TOL, TOUCH_TOL = 0.0015, 0.0008
MEM_FILE = "memory.json"

mem = {}
if os.path.exists(MEM_FILE):
    try: mem = json.load(open(MEM_FILE))
    except Exception: mem = {}

def remember(key):
    if key in mem: return False
    mem[key] = int(time.time()); return True

def send(msg):
    print("🚨", msg)
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": TG_CHAT, "text": "🚨 " + msg}, timeout=10)
        except Exception as e: print("TG ERR:", e)

def fetch_candles():
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": SYMBOL, "interval": INTERVAL, "limit": 300}, timeout=10)
    r.raise_for_status()
    return [{"t": c[0], "h": float(c[2]), "l": float(c[3]), "c": float(c[4])} for c in r.json()]

def calc_sma(closes, p): return sum(closes[-p:]) / p if len(closes) >= p else None

def calc_rsi(closes, p):
    if len(closes) < p + 1: return None
    diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    ag = sum(max(d,0) for d in diffs[-p:]) / p
    al = sum(max(-d,0) for d in diffs[-p:]) / p
    if al == 0: return 100.0
    return 100 - (100 / (1 + ag/al))

def find_pivot_levels(candles, lb):
    sup, res = [], []
    for i in range(lb, len(candles) - lb):
        w = candles[i-lb:i+lb+1]
        if candles[i]["h"] == max(c["h"] for c in w): res.append(candles[i]["h"])
        if candles[i]["l"] == min(c["l"] for c in w): sup.append(candles[i]["l"])
    return sup, res

def cluster_levels(levels, tol):
    if not levels: return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - clusters[-1][-1]) / lv <= tol: clusters[-1].append(lv)
        else: clusters.append([lv])
    return [sum(c)/len(c) for c in clusters]

def scan():
    candles = fetch_candles()
    closed  = candles[:-1]
    closes  = [k["c"] for k in closed]
    price, last_t = closes[-1], closed[-1]["t"]
    sma, rsi = calc_sma(closes, SMA_PERIOD), calc_rsi(closes, RSI_PERIOD)

    sups = cluster_levels(find_pivot_levels(closed, PIVOT_LOOKBACK)[0], LEVEL_TOL)
    ress = cluster_levels(find_pivot_levels(closed, PIVOT_LOOKBACK)[1], LEVEL_TOL)

    for s in sups:
        if abs(price-s)/price <= TOUCH_TOL and remember(f"sup_{s:.5f}_{last_t}"):
            send(f"{SYMBOL}: السعر يلمس منطقة دعم {s:.5f}")
    for r in ress:
        if abs(price-r)/price <= TOUCH_TOL and remember(f"res_{r:.5f}_{last_t}"):
            send(f"{SYMBOL}: السعر يلمس منطقة مقاومة {r:.5f}")
    if rsi is not None and rsi >= 80 and remember(f"rsih_{last_t}"):
        send(f"{SYMBOL}: RSI بتشبع شرائي ({rsi:.1f})")
    if rsi is not None and rsi <= 20 and remember(f"rsil_{last_t}"):
        send(f"{SYMBOL}: RSI بتشبع بيعي ({rsi:.1f})")
    if sma is not None:
        if closes[-2] < sma <= price and remember(f"smau_{last_t}"):
            send(f"{SYMBOL}: تقاطع فوق خط SMA{SMA_PERIOD}")
        elif closes[-2] > sma >= price and remember(f"smad_{last_t}"):
            send(f"{SYMBOL}: تقاطع تحت خط SMA{SMA_PERIOD}")

if __name__ == "__main__":
    scan()
    json.dump(mem, open(MEM_FILE, "w"))
    print("✅ اكتمل الفحص")
