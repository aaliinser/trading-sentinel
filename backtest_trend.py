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

INTERVAL = "5m"
PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0
CALM_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def get_currencies(symbol: str) -> set:
    clean = symbol.replace("=X", "")
    return {clean[:3], clean[3:6]}


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    # 1. Bollinger Bands (20) & Z-Score (20)
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    df["bb20_u"] = sma20 + 2.0 * std20
    df["bb20_l"] = sma20 - 2.0 * std20
    df["bb22_u"] = sma20 + 2.2 * std20
    df["bb22_l"] = sma20 - 2.2 * std20
    df["zscore"] = (c - sma20) / std20.replace(0.0, np.nan)

    # 2. RSI (7) & RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    ag7 = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    al7 = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    df["rsi7"] = 100.0 - (100.0 / (1.0 + ag7 / al7.replace(0.0, np.nan)))

    ag14 = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al14 = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["rsi14"] = 100.0 - (100.0 / (1.0 + ag14 / al14.replace(0.0, np.nan)))

    # 3. ADX (14) & ATR (14)
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

    # 4. CCI (14)
    tp = (h + l + c) / 3.0
    tp_sma = tp.rolling(14, min_periods=14).mean()
    mad = tp.rolling(14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["cci14"] = (tp - tp_sma) / (0.015 * mad.replace(0.0, np.nan))

    # 5. Stochastic (14, 3) & EMA (20)
    low14 = l.rolling(14, min_periods=14).min()
    high14 = h.rolling(14, min_periods=14).max()
    stoch_k = 100.0 * (c - low14) / (high14 - low14).replace(0.0, np.nan)
    df["stoch_k"] = stoch_k.rolling(3, min_periods=3).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()

    # 6. Consecutive Candles Streak (red / green)
    is_red = (c < o).astype(int)
    is_green = (c > o).astype(int)
    df["red_3"] = is_red.rolling(3).sum() == 3
    df["red_4"] = is_red.rolling(4).sum() == 4
    df["red_5"] = is_red.rolling(5).sum() == 5
    df["grn_3"] = is_green.rolling(3).sum() == 3
    df["grn_4"] = is_green.rolling(4).sum() == 4
    df["grn_5"] = is_green.rolling(5).sum() == 5

    return df


def build_15_core_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    15 نواة إشارة (1 = CALL, -1 = PUT, 0 = WAIT)
    15 نواة × 3 انتهاءات × 2 توقيت = 90 استراتيجية بالضبط
    """
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    adx = df["adx"].to_numpy()
    rsi7 = df["rsi7"].to_numpy()
    rsi14 = df["rsi14"].to_numpy()
    z = df["zscore"].to_numpy()
    cci = df["cci14"].to_numpy()
    sk = df["stoch_k"].to_numpy()
    ema = df["ema20"].to_numpy()
    atr = df["atr"].to_numpy()
    bb20_u, bb20_l = df["bb20_u"].to_numpy(), df["bb20_l"].to_numpy()
    bb22_u, bb22_l = df["bb22_u"].to_numpy(), df["bb22_l"].to_numpy()

    calm = adx < 25.0
    red = c < o
    grn = c > o

    def sig(call_m, put_m):
        out = np.zeros(len(df), dtype=np.int8)
        out[call_m] = 1
        out[put_m] = -1
        return out

    cores = {
        # عائلة 1: BB + RSI
        "F1_BB20_RSI7_15": sig(calm & red & (c < bb20_l) & (rsi7 < 15), calm & grn & (c > bb20_u) & (rsi7 > 85)),
        "F1_BB22_RSI7_20": sig(calm & red & (c < bb22_l) & (rsi7 < 20), calm & grn & (c > bb22_u) & (rsi7 > 80)),
        "F1_BB20_RSI14_25": sig(calm & red & (c < bb20_l) & (rsi14 < 25), calm & grn & (c > bb20_u) & (rsi14 > 75)),

        # عائلة 2: Z-Score Snapback
        "F2_ZScore_2.2": sig(calm & (z < -2.2) & red, calm & (z > 2.2) & grn),
        "F2_ZScore_2.5": sig(calm & (z < -2.5) & red, calm & (z > 2.5) & grn),
        "F2_ZScore_2.8": sig(calm & (z < -2.8) & red, calm & (z > 2.8) & grn),

        # عائلة 3: CCI + Bollinger
        "F3_CCI150_BB": sig(calm & (cci < -150) & (c < bb20_l), calm & (cci > 150) & (c > bb20_u)),
        "F3_CCI180_BB": sig(calm & (cci < -180) & (c < bb20_l), calm & (cci > 180) & (c > bb20_u)),
        "F3_CCI200_BB": sig(calm & (cci < -200) & (c < bb20_l), calm & (cci > 200) & (c > bb20_u)),

        # عائلة 4: Stochastic + EMA Stretch
        "F4_Stoch10_EMA": sig(calm & (sk < 10) & ((ema - c) > atr * 1.2), calm & (sk > 90) & ((c - ema) > atr * 1.2)),
        "F4_Stoch15_EMA": sig(calm & (sk < 15) & ((ema - c) > atr * 1.5), calm & (sk > 85) & ((c - ema) > atr * 1.5)),
        "F4_Stoch10_BB": sig(calm & (sk < 10) & (c < bb20_l), calm & (sk > 90) & (c > bb20_u)),

        # عائلة 5: Consecutive Candles Exhaustion
        "F5_Streak3_RSI20": sig(calm & df["red_3"].to_numpy() & (rsi7 < 20), calm & df["grn_3"].to_numpy() & (rsi7 > 80)),
        "F5_Streak4_RSI25": sig(calm & df["red_4"].to_numpy() & (rsi7 < 25), calm & df["grn_4"].to_numpy() & (rsi7 > 75)),
        "F5_Streak5_Pure": sig(calm & df["red_5"].to_numpy(), calm & df["grn_5"].to_numpy()),
    }
    return cores


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


def max_loss_streak(trades: list) -> int:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    max_l = cur_l = 0
    for t in sorted_t:
        if t["result"] == "LOSS":
            cur_l += 1
            max_l = max(max_l, cur_l)
        elif t["result"] == "WIN":
            cur_l = 0
    return max_l


def max_drawdown_usd(trades: list) -> float:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    eq = peak = mdd = 0.0
    for t in sorted_t:
        eq += t["pnl"]
        if eq > peak:
            peak = eq
        mdd = max(mdd, peak - eq)
    return round(mdd, 2)


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100.0
    logger.info("=" * 84)
    logger.info(f"90-STRATEGY DISCOVERY LAB | Interval={INTERVAL} | Period={PERIOD} | {len(PAIRS)} Non-JPY Pairs")
    logger.info("=" * 84)

    pair_cache = {}
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

        df = compute_all_indicators(df)
        cores = build_15_core_signals(df)
        pair_cache[pair] = (df, cores)
        logger.info(f"تم تجهيز {pair:<10} ({len(df)} شمعة)")

    split_ts = max(all_max_ts) - pd.Timedelta(days=20)

    stats = {}
    expiries = [(1, "5m"), (2, "10m"), (3, "15m")]
    time_modes = [("All", False), ("Calm", True)]

    for pair, (df, cores) in pair_cache.items():
        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()
        n = len(df)

        entry_ts_idx = df.index[1:]
        gaza_hours = np.asarray(entry_ts_idx.tz_convert("Asia/Gaza").hour)
        is_calm_arr = np.isin(gaza_hours, list(CALM_HOURS))
        is_train_arr = np.asarray(entry_ts_idx < split_ts)

        for core_name, sig_arr in cores.items():
            for exp_c, exp_lbl in expiries:
                valid_len = n - exp_c
                if valid_len <= 1:
                    continue

                s = sig_arr[:valid_len]
                op = opens[1 : valid_len + 1]
                cl = closes[exp_c : valid_len + exp_c]
                diff = np.where(s == 1, cl - op, np.where(s == -1, op - cl, 0.0))

                calm_slice = is_calm_arr[:valid_len]
                train_slice = is_train_arr[:valid_len]

                for t_lbl, only_calm in time_modes:
                    strat_name = f"{core_name}|{t_lbl}|{exp_lbl}"
                    if strat_name not in stats:
                        stats[strat_name] = {
                            "core": core_name, "only_calm": only_calm, "exp_c": exp_c,
                            "tr_w": 0, "tr_l": 0, "te_w": 0, "te_l": 0, "ties": 0
                        }

                    active_m = (s != 0) & (calm_slice if only_calm else True)
                    if not np.any(active_m):
                        continue

                    wins_m = active_m & (diff > 0)
                    loss_m = active_m & (diff < 0)
                    ties_m = active_m & (diff == 0)

                    st = stats[strat_name]
                    st["tr_w"] += int(np.sum(wins_m & train_slice))
                    st["tr_l"] += int(np.sum(loss_m & train_slice))
                    st["te_w"] += int(np.sum(wins_m & (~train_slice)))
                    st["te_l"] += int(np.sum(loss_m & (~train_slice)))
                    st["ties"] += int(np.sum(ties_m))

    ranking = []
    for name, st in stats.items():
        w = st["tr_w"] + st["te_w"]
        l = st["tr_l"] + st["te_l"]
        decided = w + l
        total_n = decided + st["ties"]
        if decided < 50:
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
            "core": st["core"],
            "only_calm": st["only_calm"],
            "exp_c": st["exp_c"],
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

    print("\n" + "=" * 114)
    print(f"أفضل 15 استراتيجية من أصل {len(stats)} استراتيجية على فريم 5m (خام) | نقطة التعادل: {breakeven:.2f}%")
    print("=" * 114)
    print(
        f"{'الاستراتيجية':<28} | {'صفقات':>6} | {'يومياً':>6} | {'إجمالي WR':>9} | "
        f"{'أول 40 يوم':>10} | {'آخر 20 يوم':>10} | {'الربح $':>9} | الثبات"
    )
    print("-" * 114)
    for r in ranking[:15]:
        print(
            f"{r['name']:<28} | {r['trades']:>6} | {r['daily']:>6.1f} | {r['wr']:>8.2f}% | "
            f"{r['train_wr']:>9.2f}% | {r['test_wr']:>9.2f}% | {r['pnl']:>+8.1f}$ | {r['verdict']}"
        )
    print("=" * 114)

    top5 = ranking[:5]
    cap2_results = []
    best_trades_export = []

    for idx_rank, r in enumerate(top5):
        cands = []
        exp_c = r["exp_c"]
        exp_min = exp_c * 5
        only_calm = r["only_calm"]
        core_name = r["core"]

        for pair, (df, cores) in pair_cache.items():
            sig_arr = cores[core_name]
            opens = df["open"].to_numpy()
            closes = df["close"].to_numpy()
            rsi7_arr = df["rsi7"].to_numpy()
            idx_list = df.index
            valid_len = len(df) - exp_c

            idxs = np.where(sig_arr[:valid_len] != 0)[0]
            for i in idxs:
                ts = idx_list[i + 1]
                gaza_dt = ts.tz_convert("Asia/Gaza")
                if only_calm and gaza_dt.hour not in CALM_HOURS:
                    continue

                s = int(sig_arr[i])
                op = float(opens[i + 1])
                cl = float(closes[i + exp_c])
                diff = (cl - op) if s == 1 else (op - cl)
                if diff > 0:
                    res, pnl = "WIN", STAKE * PAYOUT
                elif diff < 0:
                    res, pnl = "LOSS", -STAKE
                else:
                    res, pnl = "TIE", 0.0

                strength = abs(50.0 - float(rsi7_arr[i])) if not np.isnan(rsi7_arr[i]) else 0.0
                cands.append({
                    "pair": pair,
                    "currencies": get_currencies(pair),
                    "entry_ts": ts,
                    "exit_ts": ts + pd.Timedelta(minutes=exp_min),
                    "entry_time": str(ts),
                    "gaza_hour": gaza_dt.strftime("%H:00"),
                    "signal": "CALL" if s == 1 else "PUT",
                    "strength": round(strength, 2),
                    "result": res,
                    "pnl": round(pnl, 2),
                })

        accepted = apply_cap_2_no_overlap(cands)
        if idx_rank == 0:
            best_trades_export = accepted

        w = sum(1 for t in accepted if t["result"] == "WIN")
        l = sum(1 for t in accepted if t["result"] == "LOSS")
        wr = round((w / (w + l) * 100.0), 2) if (w + l) > 0 else 0.0

        tr_t = [t for t in accepted if t["entry_ts"] < split_ts]
        te_t = [t for t in accepted if t["entry_ts"] >= split_ts]
        tr_w = sum(1 for t in tr_t if t["result"] == "WIN")
        tr_l = sum(1 for t in tr_t if t["result"] == "LOSS")
        te_w = sum(1 for t in te_t if t["result"] == "WIN")
        te_l = sum(1 for t in te_t if t["result"] == "LOSS")
        tr_wr = round((tr_w / (tr_w + tr_l) * 100.0), 2) if (tr_w + tr_l) > 0 else 0.0
        te_wr = round((te_w / (te_w + te_l) * 100.0), 2) if (te_w + te_l) > 0 else 0.0

        pnl_sum = round(sum(t["pnl"] for t in accepted), 2)
        mdd = max_drawdown_usd(accepted)
        ml = max_loss_streak(accepted)
        stable = (tr_wr > breakeven) and (te_wr > breakeven)

        cap2_results.append({
            "name": r["name"],
            "trades": len(accepted),
            "daily": round(len(accepted) / 43.0, 1),
            "wr": wr,
            "train_wr": tr_wr,
            "test_wr": te_wr,
            "pnl_usd": pnl_sum,
            "max_loss_streak": ml,
            "max_dd_usd": mdd,
            "verdict": "ثابتة ✅" if stable else ("متذبذبة ⚠️" if wr > breakeven else "خاسرة ❌"),
        })

    print("\n" + "=" * 124)
    print(f"اختبار الحماية الصارم (Cap_2_NoOverlap) لأفضل 5 استراتيجيات على فريم 5m")
    print("=" * 124)
    print(
        f"{'الاستراتيجية':<28} | {'صفقات':>5} | {'يومياً':>6} | {'إجمالي WR':>9} | {'أول 40 يوم':>10} | "
        f"{'آخر 20 يوم':>10} | {'الربح $':>8} | {'خسارة متتالية':>12} | {'تراجع $':>8} | الثبات"
    )
    print("-" * 124)
    for c2 in cap2_results:
        print(
            f"{c2['name']:<28} | {c2['trades']:>5} | {c2['daily']:>6.1f} | {c2['wr']:>8.2f}% | "
            f"{c2['train_wr']:>9.2f}% | {c2['test_wr']:>9.2f}% | "
            f"{c2['pnl_usd']:>+7.1f}$ | {c2['max_loss_streak']:>12} | {c2['max_dd_usd']:>7.1f}$ | {c2['verdict']}"
        )
    print("=" * 124)

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"total_tested": len(stats), "interval": INTERVAL, "top15_raw": ranking[:15], "top5_cap2": cap2_results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    if best_trades_export:
        pd.DataFrame(best_trades_export).drop(columns=["entry_ts", "exit_ts", "currencies"]).to_csv("backtest_trend_trades.csv", index=False)

    print("\n✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
