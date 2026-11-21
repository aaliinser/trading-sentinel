import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# جميع أزواج الفوركس الـ 28 (21 أساسية وتقاطعية + 7 أزواج ين JPY)
PAIRS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X",
    "EURGBP=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPAUD=X", "GBPCHF=X", "GBPCAD=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
    # أزواج الين (JPY)
    "USDJPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
]

INTERVAL = "5m"
PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0
CALM_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Bollinger Bands (20, 2.0)
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    df["bb20_u"] = sma20 + 2.0 * std20
    df["bb20_l"] = sma20 - 2.0 * std20

    # Wilder RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag14 = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al14 = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["rsi14"] = 100.0 - (100.0 / (1.0 + ag14 / al14.replace(0.0, np.nan)))

    # Wilder ADX (14)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    up_m = h - h.shift(1)
    dn_m = l.shift(1) - l
    pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    mdm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    pdi = 100.0 * pd.Series(pdm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    mdi = 100.0 * pd.Series(mdm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    return df


def calc_streak_and_dd(trades: list) -> tuple[int, float]:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    max_l = cur_l = 0
    eq = peak = mdd = 0.0
    for t in sorted_t:
        if t["result"] == "LOSS":
            cur_l += 1
            max_l = max(max_l, cur_l)
        elif t["result"] == "WIN":
            cur_l = 0
        eq += t["pnl"]
        if eq > peak:
            peak = eq
        mdd = max(mdd, peak - eq)
    return max_l, round(mdd, 2)


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100.0
    logger.info("=" * 88)
    logger.info(f"M5 NO-LOCK LAB | 28 Pairs (Incl. JPY) | BB(20,2.0) + RSI(14) + ADX<25 | Period={PERIOD}")
    logger.info("=" * 88)

    configs = [
        ("RSI14_25|All|10m", 25, 75, False, 2),
        ("RSI14_25|All|15m", 25, 75, False, 3),
        ("RSI14_25|Calm|10m", 25, 75, True, 2),
        ("RSI14_28|All|10m", 28, 72, False, 2),
        ("RSI14_28|All|15m", 28, 72, False, 3),
        ("RSI14_28|Calm|10m", 28, 72, True, 2),
        ("RSI14_30|All|10m", 30, 70, False, 2),
        ("RSI14_30|All|5m", 30, 70, False, 1),
        ("RSI14_30|Calm|10m", 30, 70, True, 2),
    ]

    buckets = {cfg[0]: [] for cfg in configs}
    all_max_ts = []

    for pair in PAIRS:
        df = yf.download(pair, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if df.empty or len(df) < 50:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        all_max_ts.append(df.index.max())

        df = compute_indicators(df)
        o = df["open"].to_numpy()
        c = df["close"].to_numpy()
        bb_u = df["bb20_u"].to_numpy()
        bb_l = df["bb20_l"].to_numpy()
        rsi = df["rsi14"].to_numpy()
        adx = df["adx"].to_numpy()
        idx_list = df.index
        n = len(df)

        calm_adx = adx < 25.0
        red = c < o
        grn = c > o

        for name, r_low, r_high, only_calm, exp_c in configs:
            valid_len = n - exp_c
            if valid_len <= 1:
                continue

            call_m = calm_adx[:valid_len] & red[:valid_len] & (c[:valid_len] < bb_l[:valid_len]) & (rsi[:valid_len] < r_low)
            put_m = calm_adx[:valid_len] & grn[:valid_len] & (c[:valid_len] > bb_u[:valid_len]) & (rsi[:valid_len] > r_high)
            idxs = np.where(call_m | put_m)[0]

            for i in idxs:
                ts = idx_list[i + 1]
                gaza_dt = ts.tz_convert("Asia/Gaza")
                if only_calm and gaza_dt.hour not in CALM_HOURS:
                    continue

                sig = "CALL" if call_m[i] else "PUT"
                op = float(o[i + 1])
                cl = float(c[i + exp_c])
                diff = (cl - op) if sig == "CALL" else (op - cl)

                if diff > 0:
                    res, pnl = "WIN", STAKE * PAYOUT
                elif diff < 0:
                    res, pnl = "LOSS", -STAKE
                else:
                    res, pnl = "TIE", 0.0

                buckets[name].append({
                    "pair": pair,
                    "entry_ts": ts,
                    "entry_time": str(ts),
                    "gaza_hour": gaza_dt.strftime("%H:00"),
                    "signal": sig,
                    "result": res,
                    "pnl": round(pnl, 2),
                })

        logger.info(f"تم فحص {pair:<10} ({len(df)} شمعة)")

    split_ts = max(all_max_ts) - pd.Timedelta(days=20)
    results = []

    for name, trades in buckets.items():
        w = sum(1 for t in trades if t["result"] == "WIN")
        l = sum(1 for t in trades if t["result"] == "LOSS")
        wr = round((w / (w + l) * 100.0), 2) if (w + l) > 0 else 0.0

        tr_t = [t for t in trades if t["entry_ts"] < split_ts]
        te_t = [t for t in trades if t["entry_ts"] >= split_ts]
        tr_w = sum(1 for t in tr_t if t["result"] == "WIN")
        tr_l = sum(1 for t in tr_t if t["result"] == "LOSS")
        te_w = sum(1 for t in te_t if t["result"] == "WIN")
        te_l = sum(1 for t in te_t if t["result"] == "LOSS")

        tr_wr = round((tr_w / (tr_w + tr_l) * 100.0), 2) if (tr_w + tr_l) > 0 else 0.0
        te_wr = round((te_w / (te_w + te_l) * 100.0), 2) if (te_w + te_l) > 0 else 0.0
        pnl_sum = round(sum(t["pnl"] for t in trades), 2)
        ml, mdd = calc_streak_and_dd(trades)
        stable = (tr_wr > breakeven) and (te_wr > breakeven)

        results.append({
            "name": name,
            "trades": len(trades),
            "daily": round(len(trades) / 43.0, 1),
            "wr": wr,
            "train_wr": tr_wr,
            "test_wr": te_wr,
            "pnl_usd": pnl_sum,
            "max_loss_streak": ml,
            "max_dd_usd": mdd,
            "stable": stable,
            "verdict": "ثابتة ✅" if stable else ("متذبذبة ⚠️" if wr > breakeven else "خاسرة ❌"),
        })

    results.sort(key=lambda x: (x["stable"], x["wr"]), reverse=True)

    print("\n" + "=" * 124)
    print(f"نتائج فريم 5m بدون قفل (28 زوجاً شامل JPY) | نقطة التعادل: {breakeven:.2f}% | مخاطرة 10$ ثابتة")
    print("=" * 124)
    print(
        f"{'النسخة':<20} | {'صفقات':>5} | {'يومياً':>6} | {'إجمالي WR':>9} | {'أول 40 يوم':>10} | "
        f"{'آخر 20 يوم':>10} | {'الربح $':>8} | {'خسارة متتالية':>12} | {'تراجع $':>8} | الثبات"
    )
    print("-" * 124)
    for r in results:
        print(
            f"{r['name']:<20} | {r['trades']:>5} | {r['daily']:>6.1f} | {r['wr']:>8.2f}% | "
            f"{r['train_wr']:>9.2f}% | {r['test_wr']:>9.2f}% | "
            f"{r['pnl_usd']:>+7.1f}$ | {r['max_loss_streak']:>12} | {r['max_dd_usd']:>7.1f}$ | {r['verdict']}"
        )
    print("=" * 124)

    winner = results[0]["name"]
    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump({"pairs_count": len(PAIRS), "lock": "NONE", "winner": winner, "results": results}, f, ensure_ascii=False, indent=2)

    if buckets[winner]:
        pd.DataFrame(buckets[winner]).drop(columns=["entry_ts"]).to_csv("backtest_trend_trades.csv", index=False)

    print(f"\n🏆 الأعلى نسبة نجاح وثباتاً: {winner}")
    print("✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
