"""
MarketResearcher — Benchmark Gauntlet
=====================================

A frozen, versioned, reproducible test battery that every strategy / evolved
genome must survive before it is allowed anywhere near capital.

Design philosophy (the "ironclad" contract):

    A backtest number is worthless unless the *process* that produced it is
    immune to the four ways quant research lies to itself:

        1. Selection on the test set    -> LOCKBOX (data evolution never sees)
        2. Multiple-testing luck         -> DEFLATED SHARPE (corrected for N trials)
        3. Cost / execution fantasy      -> COST STRESS + 1-BAR-LAG FILLS
        4. Single-sample fragility       -> MULTI-ASSET + PARAM-NEIGHBOURHOOD + MONTE-CARLO

Every benchmark run emits a *scorecard* stamped with:
    - benchmark spec version   (the constitution — thresholds are frozen)
    - data manifest hash       (exact bytes of the frozen data snapshot)
    - candidate hash           (strategy + params)
    - rng seed

Two runs with identical inputs produce byte-identical scorecards. That is what
makes evolutions comparable: they are all judged by the *same* referee on the
*same* never-before-seen data.

Public API:
    from benchmark import run_gauntlet, Candidate, BENCHMARK_SPEC
"""

from benchmark.spec import BENCHMARK_SPEC, BenchmarkSpec
from benchmark.candidate import Candidate
from benchmark.scorecard import GateResult, Scorecard
from benchmark.runner import run_gauntlet

__all__ = [
    "BENCHMARK_SPEC",
    "BenchmarkSpec",
    "Candidate",
    "GateResult",
    "Scorecard",
    "run_gauntlet",
]

__version__ = "1.0.0"
