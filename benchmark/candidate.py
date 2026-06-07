"""
benchmark.candidate — what gets judged
======================================

A Candidate is a (strategy_name, params) pair plus optional provenance about the
evolution run that produced it (number of trials, spread of trial Sharpes). That
provenance feeds the Deflated Sharpe gate: a candidate cherry-picked from 30,000
evaluations must clear a far higher bar than one found in 100.

The candidate builds a `strategy_fn(df, i, pos)` that:
    - imports the strategy module from the project's STRATEGY_REGISTRY,
    - applies its params onto a DEEP COPY of the module PARAMS dict (so two
      candidates of the same strategy never contaminate each other),
    - injects the per-bar regime, matching how evolution calls strategies.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional


def _load_registry() -> dict:
    # STRATEGY_REGISTRY lives in auto_evolve.py
    mod = importlib.import_module("auto_evolve")
    return getattr(mod, "STRATEGY_REGISTRY")


@dataclass
class Candidate:
    strategy: str
    params: dict = field(default_factory=dict)
    label: str = ""
    # Provenance for the multiple-testing correction (Deflated Sharpe).
    n_trials: Optional[int] = None
    trials_sharpe_std: Optional[float] = None
    source: str = ""        # e.g. path to the genome json / evolution run id

    # ── construction helpers ──────────────────────────────────────────────────
    @classmethod
    def from_genome_file(cls, path: str) -> "Candidate":
        with open(path) as f:
            g = json.load(f)
        return cls(
            strategy=g["strategy"],
            params=g.get("params", {}),
            label=g.get("label", g["strategy"]),
            n_trials=g.get("n_trials"),
            trials_sharpe_std=g.get("trials_sharpe_std"),
            source=path,
        )

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"strategy": self.strategy, "params": self.params},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ── strategy function factory ─────────────────────────────────────────────
    def build_strategy_fn(self, param_overrides: Optional[dict] = None) -> Callable:
        """
        Returns strategy_fn(df, i, pos) with this candidate's params applied.
        `param_overrides` lets the parameter-stability gate perturb params without
        mutating the candidate.
        """
        registry = _load_registry()
        if self.strategy not in registry:
            raise KeyError(
                f"Strategy '{self.strategy}' not in STRATEGY_REGISTRY "
                f"(known: {sorted(registry)})"
            )
        reg = registry[self.strategy]
        mod = importlib.import_module(reg["module"])
        fn_ref = getattr(mod, reg["function"])
        params_attr = reg["params_dict"]

        merged = copy.deepcopy(getattr(mod, params_attr))
        merged.update(self.params)
        if param_overrides:
            merged.update(param_overrides)

        def strategy_fn(df, i, pos):
            # Apply this candidate's params just-in-time (deep-copied snapshot),
            # then restore. Single-threaded benchmark => safe and deterministic.
            original = getattr(mod, params_attr)
            setattr(mod, params_attr, merged)
            try:
                regime = "unknown"
                if "_regime" in df.columns and i < len(df):
                    regime = str(df.iloc[i].get("_regime", "unknown"))
                return fn_ref(df, i, pos, regime=regime)
            finally:
                setattr(mod, params_attr, original)

        return strategy_fn

    def numeric_param_space(self) -> dict:
        """
        Returns {name: (lo, hi)} for numeric params, taken from the registry
        param_space — used by the parameter-stability gate to perturb sanely.
        """
        registry = _load_registry()
        reg = registry.get(self.strategy, {})
        space = reg.get("param_space", {})
        out = {}
        for name, decl in space.items():
            kind = decl[0]
            if kind in ("float", "int") and len(decl) >= 3:
                out[name] = (kind, decl[1], decl[2])
        return out
