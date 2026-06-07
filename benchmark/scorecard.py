"""
benchmark.scorecard — verdict, serialisation, leaderboard
=========================================================

A Scorecard is the stamped, reproducible verdict of one gauntlet run. It records
every gate's pass/fail, the margin by which it passed/failed, the exact spec and
data the run used, and an overall PASS only if every MANDATORY gate passed.

Outputs:
    - JSON          benchmark/results/<spec_version>/<run_id>.json   (machine)
    - Markdown      printed to stdout                                (human)
    - leaderboard   benchmark/results/leaderboard.csv (append-only)  (compare runs)
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LEADERBOARD = os.path.join(RESULTS_DIR, "leaderboard.csv")


@dataclass
class GateResult:
    name: str
    passed: bool
    mandatory: bool
    weight: float = 1.0
    # margin in [-inf, +inf]; >=0 means passed with that headroom (normalised where possible)
    margin: float = 0.0
    summary: str = ""
    detail: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scorecard:
    # Identity / reproducibility stamp
    run_id: str
    built_at: str
    spec_version: str
    spec_fingerprint: str
    data_manifest_hash: str
    candidate_strategy: str
    candidate_fingerprint: str
    candidate_label: str
    seed: int

    gates: list = field(default_factory=list)        # list[GateResult]
    candidate_params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ── verdict ────────────────────────────────────────────────────────────────
    @property
    def passed(self) -> bool:
        mand = [g for g in self.gates if g.mandatory]
        return len(mand) > 0 and all(g.passed for g in mand)

    @property
    def n_passed(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    @property
    def weighted_score(self) -> float:
        """
        Aggregate quality among *passers* — used to rank ironclad candidates.
        Sum of weight*clamped_margin for passed gates, minus heavy penalty for
        any failed mandatory gate.
        """
        score = 0.0
        for g in self.gates:
            m = max(min(g.margin, 1.0), -1.0)
            if g.passed:
                score += g.weight * m
            elif g.mandatory:
                score -= g.weight * 2.0
        return round(score, 4)

    # ── serialisation ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["weighted_score"] = self.weighted_score
        d["n_gates"] = len(self.gates)
        d["n_passed"] = self.n_passed
        return d

    def save(self) -> str:
        out_dir = os.path.join(RESULTS_DIR, self.spec_version)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        self._append_leaderboard()
        return path

    def _append_leaderboard(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        new = not os.path.exists(LEADERBOARD)
        with open(LEADERBOARD, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow([
                    "run_id", "built_at", "spec_version", "spec_fp", "data_hash",
                    "strategy", "label", "candidate_fp", "passed",
                    "n_passed", "n_gates", "weighted_score",
                ])
            w.writerow([
                self.run_id, self.built_at, self.spec_version, self.spec_fingerprint,
                self.data_manifest_hash, self.candidate_strategy, self.candidate_label,
                self.candidate_fingerprint, self.passed, self.n_passed, len(self.gates),
                self.weighted_score,
            ])

    # ── human report ───────────────────────────────────────────────────────────
    def to_markdown(self) -> str:
        verdict = "✅ IRONCLAD — PASSED" if self.passed else "❌ REJECTED"
        lines = [
            f"# Benchmark Scorecard — {verdict}",
            "",
            f"- **Candidate**: `{self.candidate_strategy}` ({self.candidate_label})  fp=`{self.candidate_fingerprint}`",
            f"- **Spec**: v{self.spec_version}  fp=`{self.spec_fingerprint}`",
            f"- **Data manifest**: `{self.data_manifest_hash}`   seed=`{self.seed}`",
            f"- **Run**: `{self.run_id}`  @ {self.built_at}",
            f"- **Gates passed**: {self.n_passed}/{len(self.gates)}   weighted score: {self.weighted_score}",
            "",
            "| Gate | Mand. | Result | Margin | Summary |",
            "|------|:-----:|:------:|:------:|---------|",
        ]
        for g in self.gates:
            mark = "✅" if g.passed else ("❌" if g.mandatory else "⚠️")
            mand = "yes" if g.mandatory else "no"
            err = f" _(error: {g.error})_" if g.error else ""
            lines.append(
                f"| `{g.name}` | {mand} | {mark} | {g.margin:+.3f} | {g.summary}{err} |"
            )
        if not self.passed:
            failed = [g.name for g in self.gates if g.mandatory and not g.passed]
            lines += ["", f"**Blocking failures:** {', '.join(failed)}"]
        lines += [
            "",
            "_A candidate is IRONCLAD only if every mandatory gate passes on the "
            "frozen lockbox. Same spec + same data + same candidate ⇒ identical scorecard._",
        ]
        return "\n".join(lines)
