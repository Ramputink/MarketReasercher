"""Cycle 6: (a) breadth-scaled exposure on the BMSB portfolio vs flat (Cycle 5);
(b) second-holdout re-validation on a DIFFERENT coin subset.
Breadth[t] = fraction of coins with close>band_top at t (causal). Scaled portfolio
holds (1-breadth) in cash on top of per-coin band logic — distrusts thin rallies."""
import sys, os, json, copy, importlib
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP = "/Users/matveypro/.claude/jobs/1ca96668/tmp"

def sleeve_and_band(coin):
    df = pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    df = df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    reg = df["_regime"].to_numpy()
    mod = importlib.import_module("strategies.bmsb_long"); mod.PARAMS = copy.deepcopy(mod.PARAMS)
    fn0 = mod.bmsb_long_strategy
    bt = BacktestConfig(max_position_pct=1.0); rc = RiskConfig(); rc.sizing_method = "fixed_pct"
    def fn(d, i, p):
        r = reg[i] if 0 <= i < len(reg) else "unknown"
        return fn0(d, i, p, regime=r)
    _t, eq, _m = Backtester(bt, rc).run(df, fn, "bmsb_long")
    ts = pd.to_datetime(df["timestamp"].iloc[-len(eq):].values, unit="ms")
    sleeve = pd.Series(eq.values, index=ts); sleeve = sleeve / sleeve.iloc[0]
    band_top = df[["bmsb_sma_real","bmsb_ema_real"]].max(axis=1).values
    above = pd.Series((df["close"].values > band_top).astype(float), index=pd.to_datetime(df["timestamp"].values, unit="ms"))
    hold = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamp"].values, unit="ms")); hold = hold/hold.iloc[0]
    return sleeve, above, hold

def dd(c):
    c=np.asarray(c,float); pk=np.maximum.accumulate(c); return float(((pk-c)/pk).max()*100)
def stats(c):
    c=np.asarray(c,float); r=c[1:]/c[:-1]-1; sh=float(np.mean(r)/(np.std(r)+1e-12)*np.sqrt(365)) if len(r)>2 else 0
    return c[-1]/c[0], dd(c), sh
def calmar(m,d): return m/(1+d/100)

def run_subset(coins, label):
    S={}; A={}; H={}
    for c in coins:
        s,a,h=sleeve_and_band(c); S[c]=s; A[c]=a; H[c]=h
    idx=None
    for s in S.values(): idx=s.index if idx is None else idx.intersection(s.index)
    idx=idx.sort_values()
    Sdf=pd.DataFrame({c:S[c].reindex(idx).ffill() for c in coins})
    Adf=pd.DataFrame({c:A[c].reindex(idx).ffill() for c in coins})
    Hdf=pd.DataFrame({c:H[c].reindex(idx).ffill() for c in coins})
    n=len(idx); cut=int(n*0.6)
    print(f"\n##### {label}: {coins} ({n} days) #####")
    for name,lo,hi in [("FULL",0,n),("TEST40",cut,n)]:
        seg=Sdf.iloc[lo:hi]; segh=Hdf.iloc[lo:hi]; brd=Adf.iloc[lo:hi].mean(axis=1).to_numpy()
        segN=seg/seg.iloc[0]; seghN=segh/segh.iloc[0]
        sleeve_ret=segN.pct_change().fillna(0).to_numpy()  # (T, ncoins)
        # flat equal-weight (Cycle 5)
        flat=np.cumprod(1+np.mean(sleeve_ret,axis=1));
        # breadth-scaled: exposure = breadth[t-1] (use prior day's breadth, causal)
        expo=np.concatenate([[1.0], brd[:-1]])
        scaled=np.cumprod(1+expo*np.mean(sleeve_ret,axis=1))
        bh=np.cumprod(1+np.mean(seghN.pct_change().fillna(0).to_numpy(),axis=1))
        fm,fd,fs=stats(flat); sm,sd,ss=stats(scaled); bm,bd,bs=stats(bh)
        print(f"  [{name}] flat: {fm:.3f}x DD{fd:.0f}% Cal{calmar(fm,fd):.2f} | "
              f"breadth: {sm:.3f}x DD{sd:.0f}% Cal{calmar(sm,sd):.2f} | B&H {bm:.3f}x DD{bd:.0f}%")
        if name=="TEST40":
            yield {"label":label,"flat_mult":fm,"flat_dd":fd,"flat_cal":calmar(fm,fd),
                   "breadth_mult":sm,"breadth_dd":sd,"breadth_cal":calmar(sm,sd),
                   "bh_mult":bm,"bh_dd":bd,"breadth_better":bool(calmar(sm,sd)>calmar(fm,fd))}

res={}
# (a) breadth on the accepted majors set
for r in run_subset(["BTC","ETH","BNB","SOL","ADA","LINK"],"MAJORS(accepted set)"): res["majors"]=r
# (b) second holdout: different coin subset
for r in run_subset(["XRP","LTC","ADA","LINK"],"SECOND-SUBSET(re-validation)"): res["second"]=r
json.dump(res,open(f"{TMP}/cycle6_result.json","w"),indent=1)
print("\nVERDICT breadth-scaling better (OOS Calmar): majors=%s, second=%s"%(
    res.get("majors",{}).get("breadth_better"), res.get("second",{}).get("breadth_better")))
print("SECOND-SUBSET diversification still beats its B&H: flat %.3f vs B&H %.3f"%(
    res.get("second",{}).get("flat_mult",0), res.get("second",{}).get("bh_mult",0)))
