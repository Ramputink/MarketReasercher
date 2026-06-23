"""Cycle 7: TRUE weekly-resampled BMSB (20W SMA + 21W EMA on weekly bars, ffilled to
daily) vs the daily-140/147 approximation. Does it differ materially in return/DD?
Run bmsb_long with each band; compare per coin (FULL + recent-40%)."""
import sys, os, json, copy, importlib
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP = "/Users/matveypro/.claude/jobs/1ca96668/tmp"
COINS = ["BTC","ETH","BNB","SOL","ADA","LINK"]

def with_weekly_band(coin):
    df = pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    dt = pd.to_datetime(df["timestamp"], unit="ms")
    s = pd.Series(df["close"].values, index=dt)
    wk = s.resample("W-SUN").last()                       # weekly close
    wk_sma = wk.rolling(20).mean()                        # 20-week SMA
    wk_ema = wk.ewm(span=21, adjust=False).mean()         # 21-week EMA
    # forward-fill weekly band onto daily (causal: each day uses the last COMPLETED week)
    df["bmsb_sma_real"] = wk_sma.reindex(dt, method="ffill").values
    df["bmsb_ema_real"] = wk_ema.reindex(dt, method="ffill").values
    return df

def run(df):
    df = df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    reg = df["_regime"].to_numpy()
    mod = importlib.import_module("strategies.bmsb_long"); mod.PARAMS = copy.deepcopy(mod.PARAMS)
    fn0 = mod.bmsb_long_strategy
    bt = BacktestConfig(max_position_pct=1.0); rc = RiskConfig(); rc.sizing_method="fixed_pct"
    def fn(d,i,p):
        r = reg[i] if 0<=i<len(reg) else "unknown"
        return fn0(d,i,p,regime=r)
    out={}
    n=len(df); cut=int(n*0.6)
    for name,lo,hi in [("FULL",0,n),("TEST40",cut,n)]:
        seg=df.iloc[lo:hi].reset_index(drop=True); sreg=reg[lo:hi]
        def fn2(d,i,p):
            r=sreg[i] if 0<=i<len(sreg) else "unknown"; return fn0(d,i,p,regime=r)
        _t,eq,m=Backtester(bt,rc).run(seg,fn2,"bmsb_long")
        out[name]=(1+m.total_return_pct/100, m.max_drawdown_pct, m.total_trades)
    return out

print(f"{'coin':<6}{'window':<8}{'daily_mult':>11}{'daily_DD':>9}{'wk_mult':>9}{'wk_DD':>8}{'wk_trades':>10}")
agg={"daily_test":[], "weekly_test":[]}
for c in COINS:
    daily = pd.read_pickle(f"{TMP}/{c}_daily_feat.pkl").reset_index(drop=True)
    rd = run(daily); rw = run(with_weekly_band(c))
    for w in ["FULL","TEST40"]:
        dm,dd,_=rd[w]; wm,wd,wt=rw[w]
        print(f"{c:<6}{w:<8}{dm:>11.3f}{dd:>9.1f}{wm:>9.3f}{wd:>8.1f}{wt:>10}")
        if w=="TEST40": agg["daily_test"].append(dm); agg["weekly_test"].append(wm)
print(f"\nOOS(TEST40) mean mult: daily {np.mean(agg['daily_test']):.3f}x vs weekly {np.mean(agg['weekly_test']):.3f}x")
diff=abs(np.mean(agg['daily_test'])-np.mean(agg['weekly_test']))/np.mean(agg['daily_test'])*100
print(f"relative difference: {diff:.1f}%  -> {'MATERIAL' if diff>10 else 'immaterial (daily approx validated)'}")
json.dump({"daily_test_mean":float(np.mean(agg['daily_test'])),"weekly_test_mean":float(np.mean(agg['weekly_test'])),
           "rel_diff_pct":float(diff)}, open(f"{TMP}/cycle7_result.json","w"), indent=1)
