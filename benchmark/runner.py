"""
benchmark.runner — orchestration
=================================

Loads the frozen lockbox, builds the gate context, runs every enabled gate, and
assembles a stamped Scorecard. This is the one function evolution / CI calls.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from benchmark.spec import BENCHMARK_SPEC
from benchmark.candidate import Candidate
from benchmark.data_lockbox import DataLockbox
from benchmark.gates import GateContext, ALL_GATES
from benchmark.scorecard import Scorecard


def _run_id(candidate: Candidate, data_hash: str, spec_fp: str) -> str:
    raw = f"{candidate.fingerprint()}:{data_hash}:{spec_fp}:{candidate.strategy}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def run_gauntlet(
    candidate: Candidate,
    spec=BENCHMARK_SPEC,
    save: bool = True,
    verbose: bool = True,
) -> Scorecard:
    """
    Run the full benchmark gauntlet on a candidate and return its Scorecard.

    Reproducible: identical (candidate, spec, frozen data) ⇒ identical run_id and
    identical results. The run_id is deterministic (no timestamp), so re-running
    overwrites the same scorecard file rather than piling up duplicates.
    """
    lockbox = DataLockbox(spec)
    bar_hours = {"1h": 1.0, "4h": 4.0, "1d": 24.0, "15m": 0.25, "5m": 1/12, "1m": 1/60}.get(
        spec.timeframe, 1.0)

    ctx = GateContext(
        candidate=candidate,
        spec=spec,
        in_sample=lockbox.in_sample(),
        lockbox=lockbox.lockbox(),
        cross_lockboxes=lockbox.cross_asset_lockboxes(),
        bar_hours=bar_hours,
    )

    data_hash = lockbox.manifest_hash()
    spec_fp = spec.fingerprint()
    run_id = _run_id(candidate, data_hash, spec_fp)

    results = []
    for gate in ALL_GATES:
        gname = gate.__name__
        gcfg = spec.gates.get(gname)
        if gcfg is not None and not gcfg.enabled:
            continue
        if verbose:
            print(f"  running {gname} ...", flush=True)
        results.append(gate(ctx))

    card = Scorecard(
        run_id=run_id,
        built_at=datetime.now(timezone.utc).isoformat(),
        spec_version=spec.version,
        spec_fingerprint=spec_fp,
        data_manifest_hash=data_hash,
        candidate_strategy=candidate.strategy,
        candidate_fingerprint=candidate.fingerprint(),
        candidate_label=candidate.label or candidate.strategy,
        seed=spec.seed,
        gates=results,
        candidate_params=candidate.params,
        notes=[
            f"lockbox_cutoff_ts={lockbox.lockbox_cutoff_ts()}",
            f"in_sample_bars={len(ctx.in_sample)}",
            f"lockbox_bars={len(ctx.lockbox)}",
            f"cross_assets={list(ctx.cross_lockboxes.keys())}",
        ],
    )

    if save:
        path = card.save()
        if verbose:
            print(f"\nScorecard saved: {path}")
    return card
