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

# 21 زوج فوركس حقيقي (بدون JPY)
PAIRS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X",
    "EURGBP=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPAUD=X", "GBPCHF=X", "GBPCAD=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
]

PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0


def generate_strategy_4_signals(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Strategy 4 (RAW) - Price Action S/R + Rejection Wick (> 1.3x body)
    Wilder ATR(14) + Last 120 candles S/R (>= 3 touches)
    """
    n = len(df)
    signals = pd.Series("WAIT", index=df.index)
    if n < 120:
        return signals

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    # حساب Wilder ATR (14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atrs = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean().to_numpy(dtype=float)

    # فحص مسبق لذيل الرفض لتسريع الباك تست دون تغيير أي شرط
    body = np.abs(closes - opens)
    wick_down_arr = (np.minimum(opens, closes) - lows) > (body * 1.3)
    wick_up_arr = (highs - np.maximum(opens, closes)) > (body * 1.3)

    min_tol = 0.015 if "JPY" in symbol else 0.00015
    sig_vals = np.full(n, "WAIT", dtype=object)

    for i in range(119, n):
        w_down = wick_down_arr[i]
        w_up = wick_up_arr[i]
        if not (w_down or w_up):
            continue

        atr = atrs[i]
        if np.isnan(atr):
            continue

        tol = max(atr * 0.35, min_tol)
        win_highs = highs[i - 119 : i + 1]
        win_lows = lows[i - 119 : i + 1]

        # تجميع مستويات الدعم والمقاومة (نفس منطق الكود الخام حرفياً)
        lvl_prices = []
        lvl_touches = []

        for arr in (win_highs, win_lows):
            for price in arr:
                matched = False
                for j in range(len(lvl_prices)):
                    if abs(lvl_prices[j] - price) <= tol:
                        lvl_touches[j] += 1
                        matched = True
                        break
                if not matched:
                    lvl_prices.append(price)
                    lvl_touches.append(1)

        close_p = closes[i]
        high_p = highs[i]
        low_p = lows[i]

        valid_lvls = [lvl_prices[j] for j in range(len(lvl_prices)) if lvl_touches[j] >= 3]
        if not valid_lvls:
            continue

        near_dist = atr * 0.3

        if w_down:
            supports = sorted([p for p in valid_lvls if p <= close_p], reverse=True)[:2]
            if any(abs(low_p - s) <= near_dist for s in supports):
                sig_vals[i] = "CALL"
                continue

        if w_up:
            resistances = sorted([p for p in valid_lvls if p > close_p])[:2]
            if any(abs(high_p - r) <= near_dist for r in resistances):
                sig_vals[i] = "PUT"

    return pd.Series(sig_vals, index=df.index)


def evaluate_trades(df: pd.DataFrame, symbol: str, signals: pd.Series, expiry_candles: int) -> list:
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-expiry_candles)
    entry_time = df.index.to_series().shift(-1)

    mask = signals.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    trades = []

    for idx in df.index[mask]:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_hour = ts.tz_convert("Asia/Gaza").strftime("%H:00")

        sig = signals.loc[idx]
        op = float(entry_open.loc[idx])
        cl = float(exit_close.loc[idx])
        diff = (cl - op) if sig == "CALL" else (op - cl)

        if diff > 0:
            res, pnl = "WIN", STAKE * PAYOUT
        elif diff < 0:
            res, pnl = "LOSS", -STAKE
        else:
            res, pnl = "TIE", 0.0

        trades.append({
            "pair": symbol,
            "entry_ts": ts,
            "entry_time": str(ts),
            "gaza_hour": gaza_hour,
            "signal": sig,
            "result": res,
            "pnl": round(pnl, 2),
        })
    return trades


def calc_wr(trades_list) -> tuple:
    w = sum(1 for t in trades_list if t["result"] == "WIN")
    l = sum(1 for t in trades_list if t["result"] == "LOSS")
    decided = w + l
    wr = (w / decided * 100.0) if decided > 0 else 0.0
    return round(wr, 2), len(trades_list)


def max_loss_streak(trades) -> int:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    max_l = cur_l = 0
    for t in sorted_t:
        if t["result"] == "LOSS":
            cur_l += 1
            max_l = max(max_l, cur_l)
        elif t["result"] == "WIN":
            cur_l = 0
    return max_l


def summarize(name: str, trades: list, split_ts: pd.Timestamp, breakeven: float) -> dict:
    overall_wr, total_n = calc_wr(trades)
    edge = round(overall_wr - breakeven, 2)
    pnl = round(sum(t["pnl"] for t in trades), 2)
    ml = max_loss_streak(trades)
    daily_avg = round(total_n / 43.0, 1)

    train_wr, _ = calc_wr([t for t in trades if t["entry_ts"] < split_ts])
    test_wr, _ = calc_wr([t for t in trades if t["entry_ts"] >= split_ts])

    is_stable = (train_wr > breakeven) and (test_wr > breakeven)
    stability = "ثابتة ✅" if is_stable else ("متذبذبة ⚠️" if overall_wr > breakeven else "خاسرة ❌")

    top_pairs = []
    top_hours = []
    if trades:
        tdf = pd.DataFrame(trades)
        for pair, grp in tdf.groupby("pair"):
            w = int((grp["result"] == "WIN").sum())
            l = int((grp["result"] == "LOSS").sum())
            if (w + l) >= 15:
                top_pairs.append({
                    "pair": pair,
                    "trades": int(len(grp)),
                    "wr": round(w / (w + l) * 100, 1),
                    "pnl": round(float(grp["pnl"].sum()), 1),
                })
        top_pairs = sorted(top_pairs, key=lambda x: x["wr"], reverse=True)

        for hr, grp in tdf.groupby("gaza_hour"):
            w = int((grp["result"] == "WIN").sum())
            l = int((grp["result"] == "LOSS").sum())
            if (w + l) >= 15:
                top_hours.append({
                    "hour": hr,
                    "n": int(len(grp)),
                    "wr": round(w / (w + l) * 100, 1),
                })
        top_hours = sorted(top_hours, key=lambda x: x["wr"], reverse=True)

    return {
        "mode": name,
        "trades": total_n,
        "daily_avg": daily_avg,
        "wr": overall_wr,
        "edge": edge,
        "train_40d_wr": train_wr,
        "test_20d_wr": test_wr,
        "pnl_usd": pnl,
        "max_loss_streak": ml,
        "stability": stability,
        "top_pairs": top_pairs[:5],
        "top_hours": top_hours[:5],
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 80)
    logger.info("BACKTEST | Strategy 4 (RAW) - Price Action S/R + Rejection Wick")
    logger.info(f"Pairs={len(PAIRS)} Non-JPY | Period={PERIOD} | Payout={PAYOUT}")
    logger.info("=" * 80)

    results_trades = {
        "M5_Exp_5m (1c)": [],
        "M5_Exp_10m (2c)": [],
        "M15_Exp_15m (1c)": [],
        "M15_Exp_30m (2c)": [],
    }
    all_timestamps = []

    for interval, exp_map in [
        ("15m", [("M15_Exp_15m (1c)", 1), ("M15_Exp_30m (2c)", 2)]),
        ("5m", [("M5_Exp_5m (1c)", 1), ("M5_Exp_10m (2c)", 2)]),
    ]:
        logger.info(f"--- جاري فحص فريم {interval} ---")
        for pair in PAIRS:
            df = yf.download(pair, period=PERIOD, interval=interval, progress=False, auto_adjust=False)
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close"]].dropna()

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            all_timestamps.append(df.index.max())

            sig = generate_strategy_4_signals(df, pair)
            for label, exp_c in exp_map:
                results_trades[label].extend(evaluate_trades(df, pair, sig, exp_c))

        logger.info(f"انتهى فحص فريم {interval}")

    split_ts = max(all_timestamps) - pd.Timedelta(days=20)
    summaries = [summarize(k, v, split_ts, breakeven) for k, v in results_trades.items()]
    summaries_sorted = sorted(summaries, key=lambda x: x["wr"], reverse=True)
    best = summaries_sorted[0]

    print("\n" + "=" * 112)
    print(f"نتائج Strategy 4 (Price Action S/R - النسخة الخام) | نقطة التعادل: {breakeven:.2f}% | مخاطرة 10$ ثابتة")
    print("=" * 112)
    print(
        f"{'الوضع':<18} | {'صفقات':>6} | {'يومياً':>6} | {'إجمالي WR':>9} | {'الحافة':>7} | "
        f"{'أول 40 يوم':>10} | {'آخر 20 يوم':>10} | {'الربح $':>9} | {'خسارة متتالية':>12} | الثبات"
    )
    print("-" * 112)
    for s in summaries_sorted:
        print(
            f"{s['mode']:<18} | {s['trades']:>6} | {s['daily_avg']:>6.1f} | {s['wr']:>8.2f}% | "
            f"{s['edge']:>+6.2f}% | {s['train_40d_wr']:>9.2f}% | {s['test_20d_wr']:>9.2f}% | "
            f"{s['pnl_usd']:>+8.1f}$ | {s['max_loss_streak']:>12} | {s['stability']}"
        )
    print("=" * 112)

    print(f"\nأفضل 5 أزواج في الأعلى أداءً ({best['mode']}):")
    for p in best["top_pairs"]:
        print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  PnL={p['pnl']:>+7.1f}$")

    print(f"\nأفضل 5 ساعات (غزة) في الأعلى أداءً ({best['mode']}):")
    for h in best["top_hours"]:
        print(f"  {h['hour']}  WR={h['wr']:>5.1f}%  n={h['n']}")

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"strategy": "Strategy 4 (RAW) - Price Action S/R", "breakeven": round(breakeven, 2), "modes": summaries_sorted},
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_trades = results_trades[best["mode"]]
    if best_trades:
        pd.DataFrame(best_trades).drop(columns=["entry_ts"]).to_csv("backtest_trend_trades.csv", index=False)

    print("\n✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
