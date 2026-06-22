"""Cycle 3: does volatility-targeted sizing improve BMSB risk-adjusted return OOS?
Train (first 60%) picks vol_target_annual by Calmar; test (last 40%) compares vs flat."""
import sys, os, copy, importlib
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP = "/Users/matveypro/.claude/jobs/1ca96668/tmp"
COINS = ["BTC", "ETH", "BNB", "SOL", "ADA", "LINK"]

def load(c):
    df = pd.read_pickle(f"{TMP}/{c}_daily_feat.pkl").reset_index(drop=True)
    df = df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    return df, df["_regime"].to_numpy()

def run(df, reg, override):
    mod = importlib.import_module("strategies.bmsb_long")
    base = copy.deepcopy(mod.PARAMS); base.update(override); mod.PARAMS = base
    fn0 = mod.bmsb_long_strategy
    bt = BacktestConfig(max_position_pct=1.0); rc = RiskConfig(); rc.sizing_method = "fixed_pct"
    def fn(d, i, p):
        r = reg[i] if 0 <= i < len(reg) else "unknown"
        return fn0(d, i, p, regime=r)
    _t, _e, m = Backtester(bt, rc).run(df, fn, "bmsb_long")
    return (1 + m.total_return_pct/100), m.max_drawdown_pct, m.sharpe_ratio

def calmar(mult, dd): return mult / (1 + dd/100)

data = {c: load(c) for c in COINS}
cut = {c: int(len(data[c][0])*0.6) for c in COINS}
FLAT = {"enable_vol_target": False}
TARGETS = [0.40, 0.60, 0.80, 1.00, 1.50]

# pick best vol_target on TRAIN by mean Calmar
best_vt, best_sc = None, -1e9
for vt in TARGETS:
    ov = {"enable_vol_target": True, "vol_target_annual": vt, "vol_lookback": 30, "strength_floor": 0.2}
    scs = []
    for c in COINS:
        df, reg = data[c]; k = cut[c]
        mult, dd, _ = run(df.iloc[:k].reset_index(drop=True), reg[:k], ov)
        scs.append(calmar(mult, dd))
    s = float(np.mean(scs))
    print(f"train vol_target={vt}: mean Calmar {s:.3f}")
    if s > best_sc: best_sc, best_vt = s, vt
print(f"\nBEST train vol_target_annual = {best_vt}\n")

bestov = {"enable_vol_target": True, "vol_target_annual": best_vt, "vol_lookback": 30, "strength_floor": 0.2}
print(f"{'coin':<6}{'flat_mult':>10}{'flat_DD':>9}{'flat_Cal':>9} | {'vt_mult':>9}{'vt_DD':>8}{'vt_Cal':>8}{'better?':>9}")
fc, vc = [], []
for c in COINS:
    df, reg = data[c]; k = cut[c]
    te, tre = df.iloc[k:].reset_index(drop=True), reg[k:]
    fm, fdd, _ = run(te, tre, FLAT); vm, vdd, _ = run(te, tre, bestov)
    fcal, vcal = calmar(fm,fdd), calmar(vm,vdd)
    fc.append(fcal); vc.append(vcal)
    print(f"{c:<6}{fm:>10.3f}{fdd:>9.1f}{fcal:>9.3f} | {vm:>9.3f}{vdd:>8.1f}{vcal:>8.3f}{('YES' if vcal>fcal else 'no'):>9}")
print(f"\nOOS mean Calmar: flat {np.mean(fc):.3f} vs vol-target {np.mean(vc):.3f} | "
      f"vol-target better on {sum(1 for a,b in zip(vc,fc) if a>b)}/{len(COINS)} coins")
