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

INTERVAL = "15m"
PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0
CALM_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def get_currencies(symbol: str) -> set:
    clean = symbol.replace("=X", "")
    return {clean[:3], clean[3:6]}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. Wilder ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    df["atr"] = atr

    # 2. Wilder ADX (14)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    df["adx"] = dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    return df


def generate_strategy_4_fixed_signals(df: pd.DataFrame, symbol: str) -> tuple[pd.Series, pd.Series]:
    """
    Strategy 4 (FIXED) - Real Swing S/R + Real Pin Bar + ADX < 25
    تُرجع: (الإشارة, قوة الذيل لترتيب الأولوية في قفل التزامن)
    """
    n = len(df)
    sig_vals = np.full(n, "WAIT", dtype=object)
    strength_vals = np.zeros(n, dtype=float)

    if n < 130:
        return pd.Series(sig_vals, index=df.index), pd.Series(strength_vals, index=df.index)

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atrs = df["atr"].to_numpy(dtype=float)
    adxs = df["adx"].to_numpy(dtype=float)

    # استخراج القمم والقيعان الحقيقية (Swing High/Low بنافذة 3 يمين و 3 يسار)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)
    for k in range(3, n - 3):
        if highs[k] == np.max(highs[k - 3 : k + 4]):
            is_swing_high[k] = True
        if lows[k] == np.min(lows[k - 3 : k + 4]):
            is_swing_low[k] = True

    body = np.abs(closes - opens)
    lower_wick = np.minimum(opens, closes) - lows
    upper_wick = highs - np.maximum(opens, closes)

    min_tol = 0.015 if "JPY" in symbol else 0.00015

    for i in range(123, n):
        atr = atrs[i]
        adx = adxs[i]
        if np.isnan(atr) or np.isnan(adx) or atr <= 0 or adx >= 25.0:
            continue

        # شرط Pin Bar الحقيقي (استبعاد الدوجي الميتة)
        w_down = (lower_wick[i] >= atr * 0.6) and (lower_wick[i] >= body[i] * 1.8)
        w_up = (upper_wick[i] >= atr * 0.6) and (upper_wick[i] >= body[i] * 1.8)
        if not (w_down or w_up):
            continue

        # نأخذ فقط القمم والقيعان المؤكدة قبل الشمعة الحالية بـ 3 شموع على الأقل (منع Look-ahead)
        start_idx = i - 120
        end_idx = i - 3
        swing_prices = []
        for k in range(start_idx, end_idx):
            if is_swing_high[k]:
                swing_prices.append(highs[k])
            if is_swing_low[k]:
                swing_prices.append(lows[k])

        if len(swing_prices) < 2:
            continue

        tol = max(atr * 0.35, min_tol)
        lvl_prices = []
        lvl_touches = []

        for price in swing_prices:
            matched = False
            for j in range(len(lvl_prices)):
                if abs(lvl_prices[j] - price) <= tol:
                    lvl_touches[j] += 1
                    matched = True
                    break
            if not matched:
                lvl_prices.append(price)
                lvl_touches.append(1)

        # نشترط ارتدادين حقيقيين على الأقل (2 Swings)
        valid_lvls = [lvl_prices[j] for j in range(len(lvl_prices)) if lvl_touches[j] >= 2]
        if not valid_lvls:
            continue

        close_p = closes[i]
        high_p = highs[i]
        low_p = lows[i]
        near_dist = atr * 0.35

        if w_down:
            supports = sorted([p for p in valid_lvls if p <= close_p], reverse=True)[:2]
            if any(abs(low_p - s) <= near_dist for s in supports):
                sig_vals[i] = "CALL"
                strength_vals[i] = round(lower_wick[i] / atr, 2)
                continue

        if w_up:
            resistances = sorted([p for p in valid_lvls if p > close_p])[:2]
            if any(abs(high_p - r) <= near_dist for r in resistances):
                sig_vals[i] = "PUT"
                strength_vals[i] = round(upper_wick[i] / atr, 2)

    return pd.Series(sig_vals, index=df.index), pd.Series(strength_vals, index=df.index)


def extract_trades(
    df: pd.DataFrame,
    symbol: str,
    sig: pd.Series,
    strength: pd.Series,
    expiry_candles: int,
    allowed_hours: set | None,
) -> list:
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-expiry_candles)
    entry_time = df.index.to_series().shift(-1)
    expiry_minutes = expiry_candles * 15

    mask = sig.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    trades = []

    for idx in df.index[mask]:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_dt = ts.tz_convert("Asia/Gaza")

        if allowed_hours is not None and gaza_dt.hour not in allowed_hours:
            continue

        s = sig.loc[idx]
        op = float(entry_open.loc[idx])
        cl = float(exit_close.loc[idx])
        diff = (cl - op) if s == "CALL" else (op - cl)

        if diff > 0:
            res, pnl = "WIN", STAKE * PAYOUT
        elif diff < 0:
            res, pnl = "LOSS", -STAKE
        else:
            res, pnl = "TIE", 0.0

        trades.append({
            "pair": symbol,
            "currencies": get_currencies(symbol),
            "entry_ts": ts,
            "exit_ts": ts + pd.Timedelta(minutes=expiry_minutes),
            "entry_time": str(ts),
            "gaza_hour": gaza_dt.strftime("%H:00"),
            "signal": s,
            "strength": float(strength.loc[idx]),
            "result": res,
            "pnl": round(pnl, 2),
        })
    return trades


def apply_cap_2_no_overlap(candidates: list) -> list:
    ordered = sorted(candidates, key=lambda x: (x["entry_ts"], -x["strength"]))
    active = []
    accepted = []

    for trade in ordered:
        now = trade["entry_ts"]
        active = [a for a in active if a["exit_ts"] > now]

        if len(active) >= 2:
            continue

        if any(bool(trade["currencies"] & a["currencies"]) for a in active):
            continue

        active.append(trade)
        accepted.append(trade)

    return accepted


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


def max_drawdown_usd(trades) -> float:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    equity = peak = max_dd = 0.0
    for t in sorted_t:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def summarize(name: str, trades: list, split_ts: pd.Timestamp, breakeven: float) -> dict:
    wr, n = calc_wr(trades)
    pnl = round(sum(t["pnl"] for t in trades), 2)
    ml = max_loss_streak(trades)
    mdd = max_drawdown_usd(trades)
    daily = round(n / 43.0, 1)
    ret_dd = round(pnl / mdd, 2) if mdd > 0 else 0.0

    train_wr, _ = calc_wr([t for t in trades if t["entry_ts"] < split_ts])
    test_wr, _ = calc_wr([t for t in trades if t["entry_ts"] >= split_ts])

    is_stable = (train_wr > breakeven) and (test_wr > breakeven)
    stability = "ثابتة ✅" if is_stable else ("متذبذبة ⚠️" if wr > breakeven else "خاسرة ❌")

    return {
        "mode": name,
        "trades": n,
        "daily": daily,
        "wr": wr,
        "train_40d_wr": train_wr,
        "test_20d_wr": test_wr,
        "pnl_usd": pnl,
        "max_loss_streak": ml,
        "max_dd_usd": mdd,
        "ret_dd": ret_dd,
        "stability": stability,
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 84)
    logger.info("BACKTEST | Strategy 4 (FIXED) - Real Swing S/R + Pin Bar | M15 | Cap_2")
    logger.info("=" * 84)

    buckets = {
        "S4_AllHours_15m": [],
        "S4_AllHours_30m": [],
        "S4_CalmHours_15m": [],
        "S4_CalmHours_30m": [],
    }
    all_timestamps = []

    for pair in PAIRS:
        df = yf.download(pair, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close"]].dropna()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        all_timestamps.append(df.index.max())

        df = add_indicators(df)
        sig, strength = generate_strategy_4_fixed_signals(df, pair)

        buckets["S4_AllHours_15m"].extend(extract_trades(df, pair, sig, strength, 1, None))
        buckets["S4_AllHours_30m"].extend(extract_trades(df, pair, sig, strength, 2, None))
        buckets["S4_CalmHours_15m"].extend(extract_trades(df, pair, sig, strength, 1, CALM_HOURS))
        buckets["S4_CalmHours_30m"].extend(extract_trades(df, pair, sig, strength, 2, CALM_HOURS))

        logger.info(f"تم فحص {pair:<10} ({len(df)} شمعة)")

    split_ts = max(all_timestamps) - pd.Timedelta(days=20)
    capped = {k: apply_cap_2_no_overlap(v) for k, v in buckets.items()}
    summaries = [summarize(k, v, split_ts, breakeven) for k, v in capped.items()]
    summaries_sorted = sorted(summaries, key=lambda x: x["wr"], reverse=True)

    print("\n" + "=" * 124)
    print(f"نتائج Strategy 4 (FIXED - Real Swing S/R + Pin Bar) | نقطة التعادل: {breakeven:.2f}% | مخاطرة 10$ ثابتة")
    print("=" * 124)
    print(
        f"{'الوضع':<18} | {'صفقات':>5} | {'يومياً':>6} | {'إجمالي WR':>9} | {'أول 40 يوم':>10} | "
        f"{'آخر 20 يوم':>10} | {'الربح $':>8} | {'خسارة متتالية':>12} | {'تراجع $':>8} | {'عائد/تراجع':>9} | الثبات"
    )
    print("-" * 124)
    for s in summaries_sorted:
        print(
            f"{s['mode']:<18} | {s['trades']:>5} | {s['daily']:>6.1f} | {s['wr']:>8.2f}% | "
            f"{s['train_40d_wr']:>9.2f}% | {s['test_20d_wr']:>9.2f}% | "
            f"{s['pnl_usd']:>+7.1f}$ | {s['max_loss_streak']:>12} | "
            f"{s['max_dd_usd']:>7.1f}$ | {s['ret_dd']:>9.2f} | {s['stability']}"
        )
    print("=" * 124)

    winner = summaries_sorted[0]["mode"]
    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"strategy": "Strategy 4 (FIXED) - Real Swing S/R + Pin Bar", "breakeven": round(breakeven, 2), "results": summaries_sorted},
            f,
            ensure_ascii=False,
            indent=2,
        )

    if capped[winner]:
        pd.DataFrame(capped[winner]).drop(columns=["entry_ts", "exit_ts", "currencies"]).to_csv("backtest_trend_trades.csv", index=False)

    print(f"\n🏆 الأعلى في هذا الاختبار: {winner}")
    print("✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
