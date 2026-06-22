#!/usr/bin/env python3
"""
Red de seguridad del backtester (gate de Fase 1.3 del plan de mejora).

Captura un "golden snapshot" de las métricas que produce el backtester ACTUAL
para un conjunto fijo de genomas, y permite comparar contra él tras cualquier
optimización (p.ej. el kernel Numba). El objetivo es garantizar resultados
numéricamente idénticos: si el backtester rápido difiere, este test falla.

Uso:
    python tests/test_backtester_regression.py --snapshot   # genera el golden
    python tests/test_backtester_regression.py --check      # compara vs golden

El snapshot se guarda en tests/golden_backtester.json (versionable).
"""
import argparse
import copy
import glob
import importlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_backtester.json")
PREP_CACHE = "/tmp/crypto_evolve/bench_prepared_df_pinned.pkl"
# PINNED data snapshot: the golden harness must be reproducible and must NOT drift
# when the live OHLCV cache is refreshed (fetching advances the rolling window and
# silently breaks the golden). Falls back to the live parquet if the pin is absent.
_PINNED = "data/pinned_XRP_USDT_1h.parquet"
PARQUET = _PINNED if os.path.exists(_PINNED) else "data/binance_XRP_USDT_1h.parquet"
TOL = 1e-9  # tolerancia de igualdad numérica


def prepare_df() -> pd.DataFrame:
    if os.path.exists(PREP_CACHE):
        return pd.read_pickle(PREP_CACHE)
    from engine.features import build_all_features
    from mirofish.scenario_engine import classify_regime_quantitative
    raw = pd.read_parquet(PARQUET)
    df = build_all_features(raw)
    labels = ["unknown" if i < 50 else classify_regime_quantitative(df, bar_idx=i).regime.value
              for i in range(len(df))]
    df["_regime"] = labels
    os.makedirs(os.path.dirname(PREP_CACHE), exist_ok=True)
    df.to_pickle(PREP_CACHE)
    return df


def build_strategy_fn(strategy_name, params):
    """Replica exactamente cómo evaluate_genome construye la strategy_fn."""
    from auto_evolve import STRATEGY_REGISTRY
    reg = STRATEGY_REGISTRY[strategy_name]
    mod = importlib.import_module(reg["module"])
    fn_ref = getattr(mod, reg["function"])
    params_dict_ref = copy.deepcopy(getattr(mod, reg["params_dict"]))
    params_dict_ref.update(params)
    setattr(mod, reg["params_dict"], params_dict_ref)

    def _bar_regime(d, i):
        return d.iloc[i].get("_regime", "unknown") if i < len(d) else "unknown"

    def strategy_fn(d, i, p):
        return fn_ref(d, i, p, regime=_bar_regime(d, i))

    return strategy_fn


def run_one(df, strategy_name, params):
    from config import LabConfig
    from engine.backtester import Backtester
    cfg = LabConfig()
    bt = Backtester(cfg.backtest, cfg.risk)
    strategy_fn = build_strategy_fn(strategy_name, params)
    _, _, m = bt.run(df, strategy_fn, strategy_name)
    return {
        "sharpe": m.sharpe_ratio,
        "sortino": m.sortino_ratio,
        "pf": m.profit_factor,
        "trades": m.total_trades,
        "win_rate": m.win_rate,
        "net_pnl": m.net_pnl,
        "max_dd": m.max_drawdown_pct,
    }


def collect_cases():
    """Genomas de prueba: best_genome_*.json de estrategias aún en el registro."""
    from auto_evolve import STRATEGY_REGISTRY
    cases = []
    for path in sorted(glob.glob("reports/best_genome_*.json")):
        with open(path) as f:
            g = json.load(f)
        if "strategy" not in g or "params" not in g:
            continue
        if g["strategy"] not in STRATEGY_REGISTRY:
            print(f"  (skip {os.path.basename(path)}: estrategia '{g['strategy']}' no registrada)")
            continue
        label = os.path.basename(path).replace("best_genome_", "").replace(".json", "")
        cases.append({"label": label, "strategy": g["strategy"], "params": g["params"]})
    return cases


def do_snapshot(df):
    cases = collect_cases()
    snap = {}
    for c in cases:
        print(f"  snapshot {c['label']} ({c['strategy']}) ...")
        snap[c["label"]] = {"strategy": c["strategy"], "params": c["params"],
                            "metrics": run_one(df, c["strategy"], c["params"])}
    with open(GOLDEN, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"\n✓ Golden snapshot guardado: {GOLDEN} ({len(snap)} casos)")


def _close(a, b):
    if a is None or b is None:
        return a == b
    if isinstance(a, float) and (math.isnan(a) or math.isinf(a)):
        return repr(a) == repr(b)
    return abs(a - b) <= TOL + TOL * abs(b)


def do_check(df):
    if not os.path.exists(GOLDEN):
        print("✗ No hay golden snapshot. Corre --snapshot primero.")
        return 1
    with open(GOLDEN) as f:
        golden = json.load(f)
    failures = 0
    for label, ref in golden.items():
        got = run_one(df, ref["strategy"], ref["params"])
        diffs = [k for k in ref["metrics"] if not _close(got[k], ref["metrics"][k])]
        if diffs:
            failures += 1
            print(f"✗ {label}: difiere en {diffs}")
            for k in diffs:
                print(f"    {k}: golden={ref['metrics'][k]}  got={got[k]}")
        else:
            print(f"✓ {label}: idéntico")
    if failures:
        print(f"\n✗ {failures} casos difieren del golden.")
        return 1
    print(f"\n✓ Los {len(golden)} casos coinciden con el golden (tol={TOL}).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="Generar golden snapshot")
    ap.add_argument("--check", action="store_true", help="Comparar vs golden")
    args = ap.parse_args()
    df = prepare_df()
    if args.snapshot:
        do_snapshot(df)
    elif args.check:
        sys.exit(do_check(df))
    else:
        print("Usa --snapshot o --check")


if __name__ == "__main__":
    main()
