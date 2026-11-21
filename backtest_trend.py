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

# 21 زوج فوركس حقيقي (بدون أي زوج ين JPY)
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
CALM_HOURS_GAZA = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Bollinger Bands (20, 2.0)
    sma20 = df["close"].rolling(window=20, min_periods=20).mean()
    std20 = df["close"].rolling(window=20, min_periods=20).std()
    df["bb_upper"] = sma20 + (2.0 * std20)
    df["bb_lower"] = sma20 - (2.0 * std20)

    # 2. RSI (7) السريع لفريم 15 دقيقة
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi7"] = 100 - (100 / (1 + rs))

    # 3. ADX (14) لفلترة السوق العنيف
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # 4. قياسات الشمعة للذيل (Exhaustion Wick)
    df["body"] = (df["close"] - df["open"]).abs()
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)

    return df


def get_variant_signals(df: pd.DataFrame) -> dict:
    calm = df["adx"] < 25

    # الشرط الأساسي (A): BB(20, 2.0) + RSI(7) < 20 / > 80 + ADX < 25
    base_call = calm & (df["close"] < df["open"]) & (df["close"] < df["bb_lower"]) & (df["rsi7"] < 20)
    base_put = calm & (df["close"] > df["open"]) & (df["close"] > df["bb_upper"]) & (df["rsi7"] > 80)

    sig_a = pd.Series("WAIT", index=df.index)
    sig_a[base_call] = "CALL"
    sig_a[base_put] = "PUT"

    # النسخة (B): نفس A + شرط ذيل رفض واضح (الذيل >= 40% من جسم الشمعة)
    sig_b = pd.Series("WAIT", index=df.index)
    sig_b[base_call & (df["lower_wick"] >= df["body"] * 0.4)] = "CALL"
    sig_b[base_put & (df["upper_wick"] >= df["body"] * 0.4)] = "PUT"

    return {
        "A_Fast_15m": (sig_a, 1, False),          # انتهاء شمعة واحدة (15 دقيقة)
        "B_Wick_15m": (sig_b, 1, False),          # فلتر ذيل الرفض (15 دقيقة)
        "C_Fast_30m": (sig_a, 2, False),          # انتهاء شمعتين (30 دقيقة)
        "D_CalmHours_15m": (sig_a, 1, True),      # ساعات الهدوء بتوقيت غزة فقط
    }


def evaluate_variant(df: pd.DataFrame, symbol: str, signals: pd.Series, expiry_candles: int, filter_hours: bool):
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-expiry_candles)
    entry_time = df.index.to_series().shift(-1)

    mask = signals.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    idxs = df.index[mask]

    trades = []
    for idx in idxs:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_dt = ts.tz_convert("Asia/Gaza")

        if filter_hours and gaza_dt.hour not in CALM_HOURS_GAZA:
            continue

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
            "entry_time": str(ts),
            "gaza_hour": gaza_dt.strftime("%H:00"),
            "signal": sig,
            "result": res,
            "pnl": round(pnl, 2),
        })
    return trades


def max_loss_streak(trades) -> int:
    max_l = cur_l = 0
    for t in trades:
        if t["result"] == "LOSS":
            cur_l += 1
            max_l = max(max_l, cur_l)
        elif t["result"] == "WIN":
            cur_l = 0
    return max_l


def summarize_variant(name: str, trades: list, breakeven: float) -> dict:
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    ties = sum(1 for t in trades if t["result"] == "TIE")
    decided = wins + losses
    wr = (wins / decided * 100) if decided > 0 else 0.0
    edge = wr - breakeven
    pnl = sum(t["pnl"] for t in trades)
    ml = max_loss_streak(trades)

    # حساب أفضل الأزواج لهذه النسخة
    pair_stats = []
    if trades:
        tdf = pd.DataFrame(trades)
        for pair, grp in tdf.groupby("pair"):
            w = int((grp["result"] == "WIN").sum())
            l = int((grp["result"] == "LOSS").sum())
            if (w + l) >= 15:  # فقط الأزواج التي فيها 15 صفقة فأكثر لتجنب الحظ
                pair_stats.append({
                    "pair": pair,
                    "trades": int(len(grp)),
                    "wr": round(w / (w + l) * 100, 1),
                    "pnl": round(float(grp["pnl"].sum()), 1),
                })
        pair_stats = sorted(pair_stats, key=lambda x: x["wr"], reverse=True)

    return {
        "variant": name,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "wr": round(wr, 2),
        "edge": round(edge, 2),
        "pnl_usd": round(pnl, 2),
        "max_loss_streak": ml,
        "verdict": "رابحة ✅" if edge > 0 else "خاسرة ❌",
        "top_pairs": pair_stats[:5],
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 72)
    logger.info(f"MATRIX LAB | 4 Variants | {len(PAIRS)} Non-JPY Pairs | M15 | 60d")
    logger.info("=" * 72)

    all_variant_trades = {
        "A_Fast_15m": [],
        "B_Wick_15m": [],
        "C_Fast_30m": [],
        "D_CalmHours_15m": [],
    }

    loaded_pairs = 0
    for pair in PAIRS:
        df = yf.download(pair, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if df.empty:
            logger.warning(f"{pair:<10} لا توجد بيانات")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close"]].dropna()

        df = add_indicators(df)
        variants = get_variant_signals(df)

        for v_name, (sig_series, exp_candles, f_hours) in variants.items():
            v_trades = evaluate_variant(df, pair, sig_series, exp_candles, f_hours)
            all_variant_trades[v_name].extend(v_trades)

        loaded_pairs += 1
        logger.info(f"تم فحص {pair:<10} ({len(df)} شمعة)")

    summaries = [
        summarize_variant(v_name, trades, breakeven)
        for v_name, trades in all_variant_trades.items()
    ]
    summaries_sorted = sorted(summaries, key=lambda x: x["wr"], reverse=True)
    best = summaries_sorted[0]

    print("\n" + "=" * 78)
    print(f"جدول مقارنة المصفوفة ({loaded_pairs} زوج بدون JPY | فريم 15m | نقطة التعادل {breakeven:.2f}%)")
    print("=" * 78)
    print(f"{'النسخة':<18} | {'الصفقات':>7} | {'فوز/خسارة':>11} | {'النسبة':>7} | {'الحافة':>7} | {'الربح $':>9} | {'خسارة متتالية':>12} | الحكم")
    print("-" * 78)
    for s in summaries_sorted:
        wl = f"{s['wins']}/{s['losses']}"
        print(
            f"{s['variant']:<18} | {s['trades']:>7} | {wl:>11} | "
            f"{s['wr']:>6.2f}% | {s['edge']:>+6.2f}% | {s['pnl_usd']:>+8.1f}$ | "
            f"{s['max_loss_streak']:>12} | {s['verdict']}"
        )
    print("=" * 78)

    print(f"\nأفضل 5 أزواج في النسخة الأعلى ({best['variant']}) [حد أدنى 15 صفقة]:")
    for p in best["top_pairs"]:
        print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  PnL={p['pnl']:>+7.1f}$")

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump({"breakeven": round(breakeven, 2), "variants": summaries_sorted}, f, ensure_ascii=False, indent=2)

    best_trades = all_variant_trades[best["variant"]]
    if best_trades:
        pd.DataFrame(best_trades).to_csv("backtest_trend_trades.csv", index=False)

    print("\n✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
