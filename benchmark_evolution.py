#!/usr/bin/env python3
"""
Benchmark del motor de evolución — mide velocidad de entreno (Fase 1 del plan).

Offline: lee el parquet XRP/USDT 1h cacheado, construye features + regímenes una
sola vez (cacheado en /tmp), y cronometra N generaciones reales con el pool de
procesos persistente. Reporta gen/s, genomas/s y speedup teórico vs serie.

Uso:
    python benchmark_evolution.py --gens 5 --pop 40 --cores 0
    python benchmark_evolution.py --gens 5 --pop 40 --cores 1   # baseline serie
"""
import argparse
import os
import time

import pandas as pd

PREP_CACHE = "/tmp/crypto_evolve/bench_prepared_df.pkl"
PARQUET = "data/binance_XRP_USDT_1h.parquet"


def prepare_df() -> pd.DataFrame:
    """Construye (o recupera de cache) el df con features + regímenes."""
    if os.path.exists(PREP_CACHE):
        return pd.read_pickle(PREP_CACHE)

    from engine.features import build_all_features
    from mirofish.scenario_engine import classify_regime_quantitative

    raw = pd.read_parquet(PARQUET)
    df = build_all_features(raw)

    regime_labels = []
    for i in range(len(df)):
        if i < 50:
            regime_labels.append("unknown")
        else:
            regime_labels.append(classify_regime_quantitative(df, bar_idx=i).regime.value)
    df["_regime"] = regime_labels

    os.makedirs(os.path.dirname(PREP_CACHE), exist_ok=True)
    df.to_pickle(PREP_CACHE)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=5, help="Generaciones a cronometrar")
    parser.add_argument("--pop", type=int, default=40, help="Tamaño de población")
    parser.add_argument("--cores", type=int, default=0, help="Workers (0=auto, 1=baseline serie)")
    args = parser.parse_args()

    import multiprocessing as mp
    from auto_evolve import EvolutionEngine

    workers = args.cores if args.cores > 0 else max(1, mp.cpu_count() - 1)

    print(f"Preparando datos (cache: {PREP_CACHE}) ...")
    df = prepare_df()
    print(f"  {len(df)} barras, {len(df.columns)} columnas")
    print(f"Benchmark: pop={args.pop}, workers={workers}, gens={args.gens}\n")

    engine = EvolutionEngine(
        df=df, max_hours=999, pop_size=args.pop, max_workers=workers,
        train_days=90, val_days=15, test_days=45,
    )

    try:
        # Generación inicial (incluye arranque del pool: spawn + warm-up del df cache)
        population = engine.generate_initial_population()
        t0 = time.time()
        population = engine.evaluate_population(population)
        warmup_s = time.time() - t0
        print(f"Gen 0 (incl. arranque pool): {warmup_s:.2f}s")

        # Generaciones en régimen (pool ya caliente)
        t_loop = time.time()
        for g in range(args.gens):
            tg = time.time()
            population = engine.create_next_generation(population)
            population = engine.evaluate_population(population)
            print(f"  Gen {g+1}: {time.time()-tg:.2f}s  "
                  f"(evaluados={engine.total_evaluated}, robustos={engine.total_robust})")
        loop_s = time.time() - t_loop
    finally:
        engine.shutdown()

    per_gen = loop_s / args.gens
    per_genome = loop_s / (args.gens * args.pop)
    print("\n" + "=" * 56)
    print(f"  Generaciones cronometradas : {args.gens}")
    print(f"  Tiempo total (steady state): {loop_s:.2f}s")
    print(f"  s/generación               : {per_gen:.2f}s")
    print(f"  s/genoma                   : {per_genome:.3f}s")
    print(f"  generaciones/hora          : {3600/per_gen:.0f}")
    print(f"  genomas/hora               : {3600/per_genome:.0f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
