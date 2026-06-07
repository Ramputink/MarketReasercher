"""
benchmark CLI
=============

    python -m benchmark snapshot [--force]
        Build & freeze the data snapshot (primary + cross assets) once. Requires
        network on first run; afterwards the benchmark is offline & reproducible.

    python -m benchmark run --strategy donchian_breakout --params '{"entry_period": 41}'
    python -m benchmark run --genome reports/best_genome_lstm_pattern.json
        Run the full gauntlet on a candidate; prints the scorecard and saves JSON.
        Exit code 0 = IRONCLAD (passed), 1 = rejected. (CI-friendly.)

    python -m benchmark selftest
        Run the harness self-tests on deterministic synthetic data (no network).

    python -m benchmark leaderboard
        Print the append-only comparison table of all past runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_snapshot(args):
    from benchmark.spec import BENCHMARK_SPEC
    from benchmark.data_lockbox import build_snapshots
    print(f"Building frozen snapshot: {BENCHMARK_SPEC.primary_symbol} + "
          f"{list(BENCHMARK_SPEC.cross_symbols)} @ {BENCHMARK_SPEC.timeframe}, "
          f"{BENCHMARK_SPEC.history_days}d, lockbox={BENCHMARK_SPEC.lockbox_days}d")
    m = build_snapshots(BENCHMARK_SPEC, force_refresh=args.force)
    print(f"Frozen. manifest hash={m.fingerprint()}  lockbox_cutoff_ts={m.lockbox_cutoff_ts}")
    for sym, meta in m.symbols.items():
        print(f"  {sym:10s} {meta['n_bars']:>6} bars  sha={meta['sha256'][:12]}")


def _cmd_run(args):
    from benchmark.spec import BENCHMARK_SPEC
    from benchmark.candidate import Candidate
    from benchmark.runner import run_gauntlet

    if args.genome:
        cand = Candidate.from_genome_file(args.genome)
    elif args.strategy:
        params = json.loads(args.params) if args.params else {}
        cand = Candidate(strategy=args.strategy, params=params,
                         label=args.label or args.strategy)
    else:
        print("error: provide --genome PATH or --strategy NAME", file=sys.stderr)
        return 2

    if args.n_trials:
        cand.n_trials = args.n_trials
    if args.trials_sharpe_std:
        cand.trials_sharpe_std = args.trials_sharpe_std

    card = run_gauntlet(cand, spec=BENCHMARK_SPEC, save=not args.no_save, verbose=True)
    print("\n" + card.to_markdown())
    return 0 if card.passed else 1


def _cmd_selftest(args):
    from benchmark.selftest import run_selftests
    return 0 if run_selftests() else 1


def _cmd_leaderboard(args):
    from benchmark.scorecard import LEADERBOARD
    if not os.path.exists(LEADERBOARD):
        print("No runs yet. Run `python -m benchmark run ...` first.")
        return 0
    with open(LEADERBOARD) as f:
        print(f.read())
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchmark", description="MarketResearcher benchmark gauntlet")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("snapshot", help="build & freeze data snapshot")
    ps.add_argument("--force", action="store_true", help="re-fetch from exchange")
    ps.set_defaults(func=_cmd_snapshot)

    pr = sub.add_parser("run", help="run the gauntlet on a candidate")
    pr.add_argument("--strategy", help="strategy name from STRATEGY_REGISTRY")
    pr.add_argument("--params", help="JSON dict of param overrides")
    pr.add_argument("--genome", help="path to a genome json (overrides --strategy/--params)")
    pr.add_argument("--label", help="human label for the candidate")
    pr.add_argument("--n-trials", type=int, dest="n_trials",
                    help="number of evolution evaluations (hardens the Deflated Sharpe gate)")
    pr.add_argument("--trials-sharpe-std", type=float, dest="trials_sharpe_std",
                    help="std of trial Sharpes from the evolution run")
    pr.add_argument("--no-save", action="store_true", help="do not write scorecard/leaderboard")
    pr.set_defaults(func=_cmd_run)

    pt = sub.add_parser("selftest", help="self-test the harness on synthetic data")
    pt.set_defaults(func=_cmd_selftest)

    pl = sub.add_parser("leaderboard", help="print the comparison table")
    pl.set_defaults(func=_cmd_leaderboard)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
