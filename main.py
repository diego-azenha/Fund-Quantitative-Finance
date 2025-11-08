# main.py
import os
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sjm import run_sjm_pipeline, compute_features_from_active_returns, tune_hyperparams

DATA_FACTORS = "data_factors"
FAR_PATH = os.path.join(DATA_FACTORS, "factor_active_returns.csv")
FACTOR_RETURNS_PATH = os.path.join(DATA_FACTORS, "factor_returns_daily.csv")
SP500_PATH = os.path.join("data", "SP500.csv")

LOOKBACK_EXPANDING = True
LOOKBACK_DAYS = None
LAMBDA = 50.0
KAPPA = 9.5
TAU = 0.025
DELTA = 2.5
TARGET_TE = 0.02
SAVE_ARTIFACTS_DIR = "artifacts/sjm"

REBALANCE_FREQ = "M"
EXECUTION_LAG_DAYS = 1

PLOT_OUTPUT = "results_nav.png"
TABLE_OUTPUT = "results_table.csv"

# train / test split for your dataset (you requested validation to start in 2018)
TRAIN_START = "2005-01-01"
TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"
PRETUNE = False  # set to True to enable hyperparameter pre-tuning


def load_returns():
    fr = pd.read_csv(FACTOR_RETURNS_PATH, index_col=0, parse_dates=True)
    fr.index = pd.to_datetime(fr.index)
    fr = fr.sort_index()

    sp = None
    if os.path.exists(SP500_PATH):
        sp_raw = pd.read_csv(SP500_PATH, index_col=0, parse_dates=True).iloc[:, 0]
        sp_raw.index = pd.to_datetime(sp_raw.index)
        sp_raw = sp_raw.sort_index()
        sp_numeric = pd.to_numeric(sp_raw.astype(str).str.replace(',', '').str.strip(), errors='coerce')
        valid = sp_numeric.dropna()
        if len(valid) == 0:
            raise RuntimeError("SP500 file exists but contains no numeric values after coercion.")
        med = valid.abs().median()
        if med > 1.5:
            sp = sp_numeric.pct_change()
        else:
            sp = sp_numeric.copy()
        sp = sp.reindex(fr.index)
        sp = sp.ffill().bfill().fillna(0.0)
        sp.name = "SP500"
    else:
        sp = fr.mean(axis=1)
        sp.name = "SP500"

    fr = fr.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
    fr.index = pd.to_datetime(fr.index)
    fr = fr.reindex(sp.index)
    fr = fr.fillna(0.0)

    df_all = fr.copy()
    df_all["SP500"] = sp

    return df_all


def get_month_end_rebalances(dates_index):
    months = pd.period_range(start=dates_index.min(), end=dates_index.max(), freq='M')
    month_ends = [m.end_time.normalize() for m in months]
    trading_days = dates_index
    adjusted = []
    for me in month_ends:
        valid = trading_days[trading_days <= me]
        if len(valid) > 0:
            adjusted.append(valid[-1])
    adjusted = sorted(list(dict.fromkeys(adjusted)))
    return adjusted


def compute_metrics(returns_series):
    cum_ret = (1.0 + returns_series).cumprod().iloc[-1] - 1.0
    ann_ret = returns_series.mean() * 252
    ann_vol = returns_series.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    return {"cumulative_return": float(cum_ret), "ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}


def pre_tune_all_factors(far_path, vix_path=None, t10y2y_path=None,
                         lambda_grid=None, kappa_grid=None,
                         verbose=False):
    far = pd.read_csv(far_path, index_col=0, parse_dates=True)
    far.index = pd.to_datetime(far.index)
    far = far.sort_index()
    far_train = far.loc[:TRAIN_END]
    features = compute_features_from_active_returns(far_train, vix=None, t10y2y=None)

    # ajustar janelas de treino e validação conforme dados disponíveis
    lookback_train = 252 * 8
    total_days = len(features)
    validation_periods = min(252 * 6, max(252, total_days - lookback_train))

    tuned = {}
    lambda_grid = lambda_grid or [10.0, 25.0, 50.0, 100.0]
    kappa_grid = kappa_grid or [3.0, 6.0, 9.5, 12.0]

    factors = far_train.columns.tolist()
    os.makedirs("artifacts/sjm", exist_ok=True)
    for f in factors:
        print(f"[PreTune] Tuning factor {f} usando dados até {TRAIN_END} ...")
        try:
            lam, kap = tune_hyperparams(far_train, features, f,
                                        lambda_grid, kappa_grid,
                                        lookback_train=lookback_train,
                                        validation_periods=validation_periods,
                                        refit_freq=21,
                                        verbose=verbose)
            tuned[f] = (lam, kap)
            print(f"[PreTune] {f} -> lambda={lam}, kappa={kap}")
        except Exception as e:
            print(f"[PreTune] tuning failed for {f}: {e}; usando defaults")
            tuned[f] = (LAMBDA, KAPPA)

    joblib.dump(tuned, os.path.join("artifacts", "sjm", "tuned_params.joblib"))
    return tuned


def run_backtest():
    start_time = datetime.now()
    print(f"[Backtest] Starting backtest at {start_time.isoformat()}")

    df_returns = load_returns()
    dates = df_returns.index
    factors = [c for c in df_returns.columns if c != "SP500"]
    benchmark_assets = factors + ["SP500"]

    rebalance_dates = get_month_end_rebalances(dates)
    print(f"[Backtest] Found {len(rebalance_dates)} rebalance dates (first={rebalance_dates[0].date()}, last={rebalance_dates[-1].date()})")
    if len(rebalance_dates) < 2:
        raise RuntimeError("Not enough rebalance dates found.")

    tuned_params = None
    if PRETUNE:
        print("[Backtest] Running pre-tuning on training window (up to TRAIN_END)...")
        tuned_params = pre_tune_all_factors(FAR_PATH,
                                   lambda_grid=[10, 25, 50, 100],
                                   kappa_grid=[3, 6, 9.5, 12],
                                   verbose=True)

    rebalance_dates = [d for d in rebalance_dates if d >= pd.to_datetime(TEST_START)]
    if len(rebalance_dates) == 0:
        raise RuntimeError(f"No rebalance dates found on or after TEST_START={TEST_START}")

    weights_by_date = {}
    last_weights = pd.Series(np.ones(len(benchmark_assets)) / len(benchmark_assets), index=benchmark_assets)

    for idx, rb in enumerate(rebalance_dates, 1):
        print(f"[Backtest] ({idx}/{len(rebalance_dates)}) Rebalance date: {rb.date()}")
        lookback = None if LOOKBACK_EXPANDING else LOOKBACK_DAYS
        t0 = datetime.now()
        out = run_sjm_pipeline(
            far_path=FAR_PATH,
            vix_path=None,
            t10y2y_path=None,
            lookback_days=lookback,
            lambda_penalty=LAMBDA,
            kappa=KAPPA,
            tuned_params=tuned_params,
            tune=False,
            halflife_sigma=126,
            tau=TAU,
            delta=DELTA,
            target_te=TARGET_TE,
            save_artifacts=True,
            artifacts_dir=SAVE_ARTIFACTS_DIR,
            verbose=False
        )
        t1 = datetime.now()
        print(f"[Backtest] SJM pipeline done in {(t1 - t0).total_seconds():.1f}s")

        bl_inputs = out["bl_inputs"]
        from black_litterman import bl_pipeline
        bl_out = bl_pipeline(bl_inputs, target_te=TARGET_TE, long_only=True)
        w = bl_out["w"]

        if "SP500" not in w.index:
            w_full = w.reindex(factors).fillna(0.0)
            w_full["SP500"] = 0.0
        else:
            w_full = w.reindex(benchmark_assets).fillna(0.0)

        if w_full.sum() == 0:
            w_full = pd.Series(np.ones(len(w_full)) / len(w_full), index=w_full.index)
        else:
            w_full = w_full / w_full.sum()

        pos = dates.get_indexer_for([rb])[0]
        eff_pos = min(len(dates) - 1, pos + EXECUTION_LAG_DAYS)
        eff_date = dates[eff_pos]
        weights_by_date[eff_date] = w_full
        last_weights = w_full
        print(f"[Backtest] Weights scheduled for {eff_date.date()} (sum={w_full.sum():.4f})")

    print("[Backtest] Building daily weight matrix...")
    df_weights = pd.DataFrame(index=dates, columns=benchmark_assets, dtype=float)
    current = pd.Series(np.ones(len(benchmark_assets)) / len(benchmark_assets), index=benchmark_assets)
    for d in dates:
        if d in weights_by_date:
            current = weights_by_date[d]
        df_weights.loc[d] = current
    df_weights = df_weights.ffill().fillna(0.0)
    print("[Backtest] Daily weights ready")

    strat_daily = (df_weights * df_returns.reindex(df_weights.index)[benchmark_assets]).sum(axis=1)
    bmk_w = pd.Series(np.ones(len(benchmark_assets)) / len(benchmark_assets), index=benchmark_assets)
    benchmark_daily = (df_returns[benchmark_assets] * bmk_w).sum(axis=1)

        # NAVs
    nav_strat = (1.0 + strat_daily).cumprod()
    nav_bmk = (1.0 + benchmark_daily).cumprod()

    # cortar para começar em TEST_START (validação)
    nav_strat = nav_strat[nav_strat.index >= pd.to_datetime(TEST_START)]
    nav_bmk = nav_bmk[nav_bmk.index >= pd.to_datetime(TEST_START)]
    strat_daily = strat_daily[strat_daily.index >= pd.to_datetime(TEST_START)]
    benchmark_daily = benchmark_daily[benchmark_daily.index >= pd.to_datetime(TEST_START)]

    # métricas (somente out-of-sample)
    metrics_strat = compute_metrics(strat_daily)
    metrics_bmk = compute_metrics(benchmark_daily)

    # tabela de resultados
    results_table = pd.DataFrame({
        "strategy": metrics_strat,
        "benchmark": metrics_bmk
    })
    results_table = results_table.T

    # gráfico NAV (somente 2018+)
    plt.figure(figsize=(10, 6))
    plt.plot(nav_strat.index, nav_strat.values, label="Strategy NAV")
    plt.plot(nav_bmk.index, nav_bmk.values, label="Benchmark NAV (EW)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.title("Strategy vs Benchmark NAV (Out-of-Sample)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"[Backtest] Saved NAV plot: {PLOT_OUTPUT}")

    # salvar tabela
    results_table.to_csv(TABLE_OUTPUT)
    print(f"[Backtest] Saved results table: {TABLE_OUTPUT}")

    end_time = datetime.now()
    print(f"[Backtest] Finished at {end_time.isoformat()} (elapsed {(end_time - start_time).total_seconds():.1f}s)")

    return {
        "dates": dates,
        "nav_strategy": nav_strat,
        "nav_benchmark": nav_bmk,
        "daily_strategy": strat_daily,
        "daily_benchmark": benchmark_daily,
        "results_table": results_table
    }



if __name__ == "__main__":
    out = run_backtest()
    print("Summary:")
    print(out["results_table"])
