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

# جميع أزواج الفوركس الـ 28 (شامل أزواج الين JPY)
PAIRS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X",
    "EURGBP=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPAUD=X", "GBPCHF=X", "GBPCAD=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
    "USDJPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
]

INTERVAL = "5m"
EXPIRY_CANDLES = 1   # مدة الصفقة: 5 دقائق (شمعة واحدة)
PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0
CALM_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    # 1. Bollinger Bands (20) & Z-Score
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    df["bb20_u"] = sma20 + 2.0 * std20
    df["bb20_l"] = sma20 - 2.0 * std20
    df["bb22_u"] = sma20 + 2.2 * std20
    df["bb22_l"] = sma20 - 2.2 * std20
    df["bb25_u"] = sma20 + 2.5 * std20
    df["bb25_l"] = sma20 - 2.5 * std20
    df["zscore"] = (c - sma20) / std20.replace(0.0, np.nan)

    # 2. RSI (2, 3, 7, 14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    for p in (2, 3, 7, 14):
        ag = gain.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
        al = loss.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
        df[f"rsi{p}"] = 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, np.nan)))

    # 3. ATR (14) & ADX (14)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["atr"] = atr

    up_m = h - h.shift(1)
    dn_m = l.shift(1) - l
    pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    mdm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    pdi = 100.0 * pd.Series(pdm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    mdi = 100.0 * pd.Series(mdm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # 4. Williams %R (14) & Stochastic (5, 3)
    h14 = h.rolling(14, min_periods=14).max()
    l14 = l.rolling(14, min_periods=14).min()
    df["willr"] = -100.0 * (h14 - c) / (h14 - l14).replace(0.0, np.nan)

    h5 = h.rolling(5, min_periods=5).max()
    l5 = l.rolling(5, min_periods=5).min()
    df["stoch5"] = (100.0 * (c - l5) / (h5 - l5).replace(0.0, np.nan)).rolling(3, min_periods=3).mean()

    # 5. CCI (14)
    tp = (h + l + c) / 3.0
    tp_sma = tp.rolling(14, min_periods=14).mean()
    mad = tp.rolling(14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["cci14"] = (tp - tp_sma) / (0.015 * mad.replace(0.0, np.nan))

    # 6. Streaks & Candle Body
    is_red = (c < o).astype(int)
    is_grn = (c > o).astype(int)
    df["red_3"] = is_red.rolling(3).sum() == 3
    df["red_4"] = is_red.rolling(4).sum() == 4
    df["red_5"] = is_red.rolling(5).sum() == 5
    df["grn_3"] = is_grn.rolling(3).sum() == 3
    df["grn_4"] = is_grn.rolling(4).sum() == 4
    df["grn_5"] = is_grn.rolling(5).sum() == 5
    df["body"] = (c - o).abs()

    return df


def build_45_core_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """45 نواة إشارة × 2 توقيت (All / Calm) = 90 استراتيجية على فريم 5m بانتهاء 5m"""
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    adx = df["adx"].to_numpy()
    atr = df["atr"].to_numpy()
    body = df["body"].to_numpy()
    z = df["zscore"].to_numpy()
    cci = df["cci14"].to_numpy()
    wr = df["willr"].to_numpy()
    st5 = df["stoch5"].to_numpy()

    r2, r3, r7, r14 = df["rsi2"].to_numpy(), df["rsi3"].to_numpy(), df["rsi7"].to_numpy(), df["rsi14"].to_numpy()
    b20_u, b20_l = df["bb20_u"].to_numpy(), df["bb20_l"].to_numpy()
    b22_u, b22_l = df["bb22_u"].to_numpy(), df["bb22_l"].to_numpy()
    b25_u, b25_l = df["bb25_u"].to_numpy(), df["bb25_l"].to_numpy()

    r3_s, r4_s, r5_s = df["red_3"].to_numpy(), df["red_4"].to_numpy(), df["red_5"].to_numpy()
    g3_s, g4_s, g5_s = df["grn_3"].to_numpy(), df["grn_4"].to_numpy(), df["grn_5"].to_numpy()

    a25 = adx < 25.0
    a20 = adx < 20.0
    red = c < o
    grn = c > o
    big12 = body > (atr * 1.2)
    big15 = body > (atr * 1.5)

    def sig(cm, pm):
        out = np.zeros(len(df), dtype=np.int8)
        out[cm] = 1
        out[pm] = -1
        return out

    return {
        # --- العائلة 1: BB + RSI (9) ---
        "F1_BB20_R14_25_A25": sig(a25 & red & (c < b20_l) & (r14 < 25), a25 & grn & (c > b20_u) & (r14 > 75)),
        "F1_BB20_R14_22_A25": sig(a25 & red & (c < b20_l) & (r14 < 22), a25 & grn & (c > b20_u) & (r14 > 78)),
        "F1_BB22_R14_25_A25": sig(a25 & red & (c < b22_l) & (r14 < 25), a25 & grn & (c > b22_u) & (r14 > 75)),
        "F1_BB25_R14_28_A25": sig(a25 & red & (c < b25_l) & (r14 < 28), a25 & grn & (c > b25_u) & (r14 > 72)),
        "F1_BB20_R14_25_A20": sig(a20 & red & (c < b20_l) & (r14 < 25), a20 & grn & (c > b20_u) & (r14 > 75)),
        "F1_BB20_R7_15_A25":  sig(a25 & red & (c < b20_l) & (r7 < 15),  a25 & grn & (c > b20_u) & (r7 > 85)),
        "F1_BB22_R7_15_A25":  sig(a25 & red & (c < b22_l) & (r7 < 15),  a25 & grn & (c > b22_u) & (r7 > 85)),
        "F1_BB25_R7_20_A25":  sig(a25 & red & (c < b25_l) & (r7 < 20),  a25 & grn & (c > b25_u) & (r7 > 80)),
        "F1_BB22_R7_12_NoADX": sig(red & (c < b22_l) & (r7 < 12),      grn & (c > b22_u) & (r7 > 88)),

        # --- العائلة 2: RSI(2) & RSI(3) فائق السرعة (9) ---
        "F2_R2_5_BB20_A25":   sig(a25 & red & (c < b20_l) & (r2 < 5),   a25 & grn & (c > b20_u) & (r2 > 95)),
        "F2_R2_3_BB20_A25":   sig(a25 & red & (c < b20_l) & (r2 < 3),   a25 & grn & (c > b20_u) & (r2 > 97)),
        "F2_R2_5_BB22_A25":   sig(a25 & red & (c < b22_l) & (r2 < 5),   a25 & grn & (c > b22_u) & (r2 > 95)),
        "F2_R2_2_NoADX":      sig(red & (c < b20_l) & (r2 < 2),         grn & (c > b20_u) & (r2 > 98)),
        "F2_R3_8_BB20_A25":   sig(a25 & red & (c < b20_l) & (r3 < 8),   a25 & grn & (c > b20_u) & (r3 > 92)),
        "F2_R3_5_BB20_A25":   sig(a25 & red & (c < b20_l) & (r3 < 5),   a25 & grn & (c > b20_u) & (r3 > 95)),
        "F2_R3_8_BB22_A25":   sig(a25 & red & (c < b22_l) & (r3 < 8),   a25 & grn & (c > b22_u) & (r3 > 92)),
        "F2_R2_5_BigBody":    sig(a25 & red & big12 & (r2 < 5),         a25 & grn & big12 & (r2 > 95)),
        "F2_R3_5_A20":        sig(a20 & red & (c < b20_l) & (r3 < 5),   a20 & grn & (c > b20_u) & (r3 > 95)),

        # --- العائلة 3: Williams %R & Stochastic(5,3) (9) ---
        "F3_WR95_BB20_A25":   sig(a25 & red & (wr < -95) & (c < b20_l), a25 & grn & (wr > -5) & (c > b20_u)),
        "F3_WR98_BB20_A25":   sig(a25 & red & (wr < -98) & (c < b20_l), a25 & grn & (wr > -2) & (c > b20_u)),
        "F3_WR95_BB22_A25":   sig(a25 & red & (wr < -95) & (c < b22_l), a25 & grn & (wr > -5) & (c > b22_u)),
        "F3_WR95_R14_25":     sig(a25 & red & (wr < -95) & (r14 < 25),  a25 & grn & (wr > -5) & (r14 > 75)),
        "F3_St5_10_BB20_A25": sig(a25 & red & (st5 < 10) & (c < b20_l), a25 & grn & (st5 > 90) & (c > b20_u)),
        "F3_St5_5_BB20_A25":  sig(a25 & red & (st5 < 5) & (c < b20_l),  a25 & grn & (st5 > 95) & (c > b20_u)),
        "F3_St5_10_BB22_A25": sig(a25 & red & (st5 < 10) & (c < b22_l), a25 & grn & (st5 > 90) & (c > b22_u)),
        "F3_St5_10_R14_25":   sig(a25 & red & (st5 < 10) & (r14 < 25),  a25 & grn & (st5 > 90) & (r14 > 75)),
        "F3_WR95_St5_10":     sig(a25 & red & (wr < -95) & (st5 < 10),  a25 & grn & (wr > -5) & (st5 > 90)),

        # --- العائلة 4: Z-Score + شموع التمدد (9) ---
        "F4_Z2.2_A25":        sig(a25 & red & (z < -2.2),               a25 & grn & (z > 2.2)),
        "F4_Z2.5_A25":        sig(a25 & red & (z < -2.5),               a25 & grn & (z > 2.5)),
        "F4_Z2.8_A25":        sig(a25 & red & (z < -2.8),               a25 & grn & (z > 2.8)),
        "F4_Z2.2_Big12":      sig(a25 & red & (z < -2.2) & big12,       a25 & grn & (z > 2.2) & big12),
        "F4_Z2.5_Big12":      sig(a25 & red & (z < -2.5) & big12,       a25 & grn & (z > 2.5) & big12),
        "F4_Z2.2_Big15":      sig(a25 & red & (z < -2.2) & big15,       a25 & grn & (z > 2.2) & big15),
        "F4_Z2.5_R14_25":     sig(a25 & red & (z < -2.5) & (r14 < 25),  a25 & grn & (z > 2.5) & (r14 > 75)),
        "F4_Z2.2_R7_15":      sig(a25 & red & (z < -2.2) & (r7 < 15),   a25 & grn & (z > 2.2) & (r7 > 85)),
        "F4_Z2.5_A20":        sig(a20 & red & (z < -2.5),               a20 & grn & (z > 2.5)),

        # --- العائلة 5: إنهاك الشموع المتتالية + CCI (9) ---
        "F5_Strk3_R14_25":    sig(a25 & r3_s & (r14 < 25),              a25 & g3_s & (r14 > 75)),
        "F5_Strk4_R14_28":    sig(a25 & r4_s & (r14 < 28),              a25 & g4_s & (r14 > 72)),
        "F5_Strk4_BB20":      sig(a25 & r4_s & (c < b20_l),             a25 & g4_s & (c > b20_u)),
        "F5_Strk5_BB20":      sig(a25 & r5_s & (c < b20_l),             a25 & g5_s & (c > b20_u)),
        "F5_Strk5_A25":       sig(a25 & r5_s,                           a25 & g5_s),
        "F5_CCI180_BB20":     sig(a25 & red & (cci < -180) & (c < b20_l), a25 & grn & (cci > 180) & (c > b20_u)),
        "F5_CCI200_BB20":     sig(a25 & red & (cci < -200) & (c < b20_l), a25 & grn & (cci > 200) & (c > b20_u)),
        "F5_CCI220_BB20":     sig(a25 & red & (cci < -220) & (c < b20_l), a25 & grn & (cci > 220) & (c > b20_u)),
        "F5_CCI180_R14_25":   sig(a25 & red & (cci < -180) & (r14 < 25),  a25 & grn & (cci > 180) & (r14 > 75)),
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100.0
    logger.info("=" * 88)
    logger.info(f"90-STRATEGY M5 -> 5m (1-Candle) LAB | 28 Pairs (Incl. JPY) | No Lock | Period={PERIOD}")
    logger.info("=" * 88)

    stats = {}
    time_modes = [("All", False), ("Calm", True)]
    all_max_ts = []
    pair_cache = {}

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

        df = compute_all_indicators(df)
        cores = build_45_core_signals(df)
        pair_cache[pair] = (df, cores)
        logger.info(f"تم تجهيز {pair:<10} ({len(df)} شمعة)")

    split_ts = max(all_max_ts) - pd.Timedelta(days=20)

    for pair, (df, cores) in pair_cache.items():
        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()
        valid_len = len(df) - EXPIRY_CANDLES
        if valid_len <= 1:
            continue

        entry_ts_idx = df.index[1 : valid_len + 1]
        gaza_hours = np.asarray(entry_ts_idx.tz_convert("Asia/Gaza").hour)
        is_calm_arr = np.isin(gaza_hours, list(CALM_HOURS))
        is_train_arr = np.asarray(entry_ts_idx < split_ts)

        op = opens[1 : valid_len + 1]
        cl = closes[EXPIRY_CANDLES : valid_len + EXPIRY_CANDLES]

        for core_name, sig_arr in cores.items():
            s = sig_arr[:valid_len]
            diff = np.where(s == 1, cl - op, np.where(s == -1, op - cl, 0.0))

            for t_lbl, only_calm in time_modes:
                strat_name = f"{core_name}|{t_lbl}|5m"
                if strat_name not in stats:
                    stats[strat_name] = {"tr_w": 0, "tr_l": 0, "te_w": 0, "te_l": 0, "ties": 0}

                active_m = (s != 0) & (is_calm_arr if only_calm else True)
                if not np.any(active_m):
                    continue

                wins_m = active_m & (diff > 0)
                loss_m = active_m & (diff < 0)
                ties_m = active_m & (diff == 0)

                st = stats[strat_name]
                st["tr_w"] += int(np.sum(wins_m & is_train_arr))
                st["tr_l"] += int(np.sum(loss_m & is_train_arr))
                st["te_w"] += int(np.sum(wins_m & (~is_train_arr)))
                st["te_l"] += int(np.sum(loss_m & (~is_train_arr)))
                st["ties"] += int(np.sum(ties_m))

    ranking = []
    for name, st in stats.items():
        w = st["tr_w"] + st["te_w"]
        l = st["tr_l"] + st["te_l"]
        decided = w + l
        total_n = decided + st["ties"]
        if decided < 60:
            continue

        wr = (w / decided) * 100.0
        tr_dec = st["tr_w"] + st["tr_l"]
        te_dec = st["te_w"] + st["te_l"]
        tr_wr = (st["tr_w"] / tr_dec * 100.0) if tr_dec > 0 else 0.0
        te_wr = (st["te_w"] / te_dec * 100.0) if te_dec > 0 else 0.0
        pnl = w * (STAKE * PAYOUT) - l * STAKE
        stable = (tr_wr > breakeven) and (te_wr > breakeven)

        ranking.append({
            "name": name,
            "trades": total_n,
            "daily": round(total_n / 43.0, 1),
            "wr": round(wr, 2),
            "train_wr": round(tr_wr, 2),
            "test_wr": round(te_wr, 2),
            "pnl": round(pnl, 1),
            "stable": stable,
            "verdict": "ثابتة ✅" if stable else ("متذبذبة ⚠️" if wr > breakeven else "خاسرة ❌"),
        })

    ranking.sort(key=lambda x: (x["stable"], x["wr"]), reverse=True)

    print("\n" + "=" * 116)
    print(f"أفضل 20 استراتيجية من أصل {len(stats)} استراتيجية (فريم 5m -> انتهاء 5m) | 28 زوجاً | نقطة التعادل: {breakeven:.2f}%")
    print("=" * 116)
    print(
        f"{'الاستراتيجية':<30} | {'صفقات':>6} | {'يومياً':>6} | {'إجمالي WR':>9} | "
        f"{'أول 40 يوم':>10} | {'آخر 20 يوم':>10} | {'الربح $':>9} | الثبات"
    )
    print("-" * 116)
    for r in ranking[:20]:
        print(
            f"{r['name']:<30} | {r['trades']:>6} | {r['daily']:>6.1f} | {r['wr']:>8.2f}% | "
            f"{r['train_wr']:>9.2f}% | {r['test_wr']:>9.2f}% | {r['pnl']:>+8.1f}$ | {r['verdict']}"
        )
    print("=" * 116)

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"total_tested": len(stats), "interval": "5m", "expiry": "5m", "pairs": len(PAIRS), "top20": ranking[:20]},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n🏆 المتصدرة على 5m -> 5m: {ranking[0]['name'] if ranking else 'NONE'}")
    print("✅ حُفظ: backtest_trend_result.json")


if __name__ == "__main__":
    main()
