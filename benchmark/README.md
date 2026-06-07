# Benchmark Gauntlet

A **frozen, versioned, reproducible** test battery. Every evolved genome is judged
by the *same referee* on the *same never-before-seen data*, so results across
evolution runs are directly comparable. A candidate is declared **IRONCLAD** only
if it survives all eleven gates on the lockbox.

> Same `(candidate, spec, frozen data)` ⇒ byte-identical scorecard. That is the
> whole point: the benchmark is a fixed yardstick, not a moving one.

---

## Why this exists

The research stack (`auto_evolve.py` + `engine/backtester.py`) optimises a fitness
that *includes the out-of-sample folds*, evaluates ~30k genomes on a single XRP
series, charges a flat 5 bps, and fills at the signal-bar close. That pipeline can
(and did) crown genomes with Sharpe 3–4 and 135% drawdowns labelled "robust".

The gauntlet is the antidote. It is deliberately independent of the research
backtester and attacks the four ways a backtest lies:

| Failure mode | Defence in the gauntlet |
|---|---|
| Selection on the test set | **Lockbox** — final 120 days, never given to evolution |
| Multiple-testing luck | **Deflated Sharpe** — corrected for N trials (G03) |
| Cost / execution fantasy | **Cost stress** + **1-bar-lag fills** (G05, G06) |
| Single-sample fragility | **Multi-asset** + **param neighbourhood** + **Monte Carlo** (G08, G09, G07) |

---

## The eleven gates

| # | Gate | What it proves | Mandatory |
|---|------|----------------|:---:|
| G01 | `lockbox_performance` | Works at all on never-seen data (Sharpe, PF, trades) | ✅ |
| G02 | `oos_degradation` | Edge survived leaving the training set (≤50% decay) | ✅ |
| G03 | `deflated_sharpe` | Distinguishable from the best of N noise trials (**DSR ≥ 0.95**) | ✅ |
| G04 | `no_ruin_drawdown` | Full-sample DD ≤ 25%, account never blown | ✅ |
| G05 | `cost_stress` | Survives 4× slippage + 1.5× fees; reports breakeven bps | ✅ |
| G06 | `execution_lag` | Survives realistic next-bar-open fills (not same-bar) | ✅ |
| G07 | `monte_carlo` | Bootstrap: P(ruin) ≤ 5%, P(profit) ≥ 85% | ✅ |
| G08 | `multi_asset` | Same params generalise to ≥2 other coins | ✅ |
| G09 | `param_stability` | Robust basin, not a knife-edge optimum (≥60% of ±10% neighbours profitable) | ✅ |
| G10 | `significance` | Mean trade return t-stat ≥ 2.0 | ✅ |
| G11 | `concentration` | Not one lucky trade / one lucky regime | ⚠️ (reported, non-blocking) |

Thresholds live in `spec.py` and are **frozen**. Changing any one **requires
bumping `SPEC_VERSION`**, because it makes old scorecards incomparable. The version
travels inside every scorecard.

---

## Usage

```bash
# 1) Build & freeze the data snapshot ONCE (needs network the first time).
#    Afterwards the benchmark runs fully offline and reproducibly.
python -m benchmark snapshot

# 2) Self-test the referee (no network) — run this any time you touch the harness.
python -m benchmark selftest

# 3) Judge a candidate. Exit code 0 = IRONCLAD, 1 = rejected (CI-friendly).
python -m benchmark run --genome reports/best_genome_lstm_pattern.json \
                        --n-trials 31000 --trials-sharpe-std 1.0

python -m benchmark run --strategy donchian_breakout \
                        --params '{"entry_period": 41, "exit_period": 10}'

# 4) Compare every run ever made on this spec.
python -m benchmark leaderboard
```

`--n-trials` and `--trials-sharpe-std` come from the evolution run that produced
the genome. **They harden G03**: a genome cherry-picked from 31,000 evaluations
must clear a far higher bar than one found in 100. Pass them honestly — under-
reporting trials is the one knob that lets overfitting back in.

---

## The evolution contract (read this)

The lockbox only works if evolution never trains on it. One line does the whole job:

```python
# in auto_evolve.py, instead of loading the full 365-day series:
from benchmark.spec import BENCHMARK_SPEC
from benchmark.data_lockbox import get_evolution_window

df = get_evolution_window(BENCHMARK_SPEC)   # in-sample ONLY; lockbox stays sealed
```

`get_evolution_window` returns only bars **before** `lockbox_cutoff_ts`. The cutoff
timestamp is published in the snapshot manifest, so an evolution run can assert it
never reads past it. After evolution finishes, feed its best genome to the gauntlet —
the lockbox is genuinely out-of-sample because the GA's fitness never touched it.

---

## Architecture

```
benchmark/
├── spec.py            # frozen thresholds + SPEC_VERSION (the constitution)
├── data_lockbox.py    # freeze/load/hash snapshots; in-sample vs lockbox split; synthetic gen
├── harness.py         # ConservativeBacktester: next-bar fills, funding, ruin floor, balanced books
├── statistics.py      # Deflated/Probabilistic Sharpe, expected-max-Sharpe, t-stat (no scipy)
├── candidate.py       # (strategy, params) + provenance; builds isolated strategy_fn
├── gates.py           # the 11 gates
├── scorecard.py       # GateResult, Scorecard, JSON + Markdown + leaderboard
├── runner.py          # run_gauntlet() orchestration
├── selftest.py        # proves the referee is honest (run offline)
├── __main__.py        # CLI
├── snapshots/         # frozen parquet + manifest.json (created by `snapshot`)
└── results/           # scorecards/<spec_version>/<run_id>.json + leaderboard.csv
```

### Why a separate backtester?

A benchmark must be independent of the thing it judges. `harness.py` deliberately
does **not** reuse `engine/backtester.py`. It is stricter on every axis:

- **Next-bar-open fills** (`entry_lag_bars=1`) — kills the signal-bar-close look-ahead.
- **Worst-case intrabar ordering** — if a bar touches both stop and target, the stop wins.
- **Explicit perp funding** — an always-on cost; ignoring it overstates crypto edges.
- **Hard ruin floor** — equity crossing zero halts trading; no fictional >100% drawdowns.
- **Balanced books** — every cost (entry+exit commission, funding) is booked into the
  trade, so `Σ pnl_net == equity change` exactly. Verified by self-test T3.

The referee's correctness is itself tested (`selftest.py`): determinism, no
look-ahead, balanced books, closed-form PnL, cost monotonicity, ruin detection,
and the statistics. **If the self-tests fail, no scorecard can be trusted.**

---

## Reading a scorecard

`margin` is normalised so `≥ 0` means "passed with this much headroom"; the more
positive, the more comfortably it cleared. `weighted_score` ranks candidates that
*all* pass (higher = more robust), and heavily penalises any mandatory failure.

A green board is meant to be **rare**. If everything passes, the gauntlet is too
soft — tighten `spec.py` and bump the version. Never relax a threshold to make a
favourite genome pass; that is exactly the self-deception this package exists to stop.
