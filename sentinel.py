import os, json, time, requests

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT  = os.environ.get("TG_CHAT", "")

TIMEFRAME, RANGE = "30m", "7d"
TRADE_NOTE = "فريم 30د / صفقة 15د"
RSI_PERIOD, PIVOT_LOOKBACK = 14, 6
LEVEL_TOL, NEAR_TOL = 0.0015, 0.0012
MIN_SCORE = 3
COOLDOWN_SEC = 4 * 3600
MEM_FILE = "memory.json"

FOREX_PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X","USDCAD=X","NZDUSD=X",
    "EURGBP=X","EURJPY=X","EURCHF=X","EURAUD=X","EURCAD=X","EURNZD=X",
    "GBPJPY=X","GBPCHF=X","GBPAUD=X","GBPCAD=X","GBPNZD=X",
    "AUDJPY=X","AUDCAD=X","AUDCHF=X","AUDNZD=X",
    "CADJPY=X","CADCHF=X","NZDJPY=X","NZDCAD=X","NZDCHF=X","CHFJPY=X",
    "USDTRY=X","USDMXN=X","USDZAR=X","USDSGD=X","USDSEK=X","USDNOK=X","USDCNH=X",
]

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

def fetch_candles(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"interval": TIMEFRAME, "range": RANGE},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    ts, q = d["timestamp"], d["indicators"]["quote"][0]
    out = []
    for i in range(len(ts)):
        if None in (q["open"][i], q["high"][i], q["low"][i], q["close"][i]): continue
        out.append({"t": ts[i]*1000, "o": q["open"][i], "h": q["high"][i],
                    "l": q["low"][i], "c": q["close"][i]})
    return out

def sma_at(closes, p, ago=0):
    e = len(closes) - ago
    return sum(closes[e-p:e]) / p if e >= p else None

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

def scan_pair(symbol):
    name = symbol.replace("=X", "")
    candles = fetch_candles(symbol)
    if len(candles) < 120: return 0
    closed = candles[:-1]
    last_t = closed[-1]["t"]
    if mem.get(f"last_{name}") == last_t: return 0
    mem[f"last_{name}"] = last_t

    closes = [x["c"] for x in closed]
    o, h, l, c = closed[-1]["o"], closed[-1]["h"], closed[-1]["l"], closed[-1]["c"]
    sma35, sma50, sma100 = sma_at(closes,35), sma_at(closes,50), sma_at(closes,100)
    rsi, rsi_prev = calc_rsi(closes, RSI_PERIOD), calc_rsi(closes[:-1], RSI_PERIOD)
    if None in (sma35, sma50, sma100, rsi, rsi_prev): return 0

    sups = cluster_levels(find_pivot_levels(closed[-200:], PIVOT_LOOKBACK)[0], LEVEL_TOL)
    ress = cluster_levels(find_pivot_levels(closed[-200:], PIVOT_LOOKBACK)[1], LEVEL_TOL)

    rng = (h - l) or 1e-9
    body = c - o

    cs, cr = 0, []
    if sma50 > sma100: cs += 1; cr.append("ترند أعلى صاعد")
    if l <= sma35 <= c: cs += 1; cr.append("ارتداد من SMA35")
    if any(l <= s*(1+NEAR_TOL) and c > s for s in sups[-4:]): cs += 1; cr.append("ارتداد من دعم")
    if body > 0 and body >= 0.5*rng: cs += 1; cr.append("شمعة صاعدة قوية")
    if (rsi_prev < 30 <= rsi) or (45 <= rsi <= 65): cs += 1; cr.append("RSI صحية للصعود")

    ps, pr = 0, []
    if sma50 < sma100: ps += 1; pr.append("ترند أعلى هابط")
    if h >= sma35 >= c: ps += 1; pr.append("رفض من SMA35")
    if any(h >= r*(1-NEAR_TOL) and c < r for r in ress[-4:]): ps += 1; pr.append("رفض من مقاومة")
    if body < 0 and -body >= 0.5*rng: ps += 1; cr_note = "شمعة هابطة قوية"; pr.append(cr_note)
    if (rsi_prev > 70 >= rsi) or (35 <= rsi <= 55): ps += 1; pr.append("RSI صحية للهبوط")

    if cs == ps: return 0
    side, score, reasons = ("📈 CALL", cs, cr) if cs > ps else ("📉 PUT", ps, pr)
    if score < MIN_SCORE: return 0
    if time.time() - mem.get(f"cd_{name}", 0) < COOLDOWN_SEC: return 0
    mem[f"cd_{name}"] = int(time.time())
    send(f"🎯 قناص | {name} | {side} | ⭐ {score}/5 | {' + '.join(reasons)} | ⏱ {TRADE_NOTE}")
    return 1

if __name__ == "__main__":
    ok, fail, sent = 0, 0, 0
    for pair in FOREX_PAIRS:
        try:
            sent += scan_pair(pair); ok += 1
        except Exception as e:
            fail += 1; print("⚠️", pair, type(e).__name__, e)
        time.sleep(0.8)
    json.dump(mem, open(MEM_FILE, "w"))
    print(f"✅ فحص: {ok} تمام / {fail} تجاوز / 🎯 إشارات قناص: {sent}")
