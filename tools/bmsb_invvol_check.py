"""Cycle 11: inverse-vol (equal-RISK) sleeve weighting vs equal-weight, on the breadth
BMSB portfolio, OOS, on majors AND a 2nd subset. ACCEPT only if it beats equal-weight
OOS Calmar in BOTH sets."""
import sys, os, json, copy, importlib
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP="/Users/matveypro/.claude/jobs/1ca96668/tmp"

def sleeve_band_ret(coin):
    df=pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    df=df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    reg=df["_regime"].to_numpy()
    mod=importlib.import_module("strategies.bmsb_long"); mod.PARAMS=copy.deepcopy(mod.PARAMS)
    fn0=mod.bmsb_long_strategy
    bt=BacktestConfig(max_position_pct=1.0); rc=RiskConfig(); rc.sizing_method="fixed_pct"
    def fn(d,i,p):
        r=reg[i] if 0<=i<len(reg) else "unknown"; return fn0(d,i,p,regime=r)
    _t,eq,_m=Backtester(bt,rc).run(df,fn,"bmsb_long")
    ts=pd.to_datetime(df["timestamp"].iloc[-len(eq):].values,unit="ms")
    sl=pd.Series(eq.values,index=ts); sl=sl/sl.iloc[0]
    bt_top=df[["bmsb_sma_real","bmsb_ema_real"]].max(axis=1).values
    ab=pd.Series((df["close"].values>bt_top).astype(float),index=pd.to_datetime(df["timestamp"].values,unit="ms"))
    # coin realized vol (30d) from close, for inverse-vol weighting (causal)
    cl=pd.Series(df["close"].values,index=pd.to_datetime(df["timestamp"].values,unit="ms"))
    rv=cl.pct_change().rolling(30).std()
    return sl,ab,rv

def dd(c): c=np.asarray(c,float); pk=np.maximum.accumulate(c); return float(((pk-c)/pk).max()*100)
def cal(m,d): return m/(1+d/100)

def run(coins,label):
    S={};A={};V={}
    for c in coins:
        s,a,v=sleeve_band_ret(c); S[c]=s;A[c]=a;V[c]=v
    idx=None
    for s in S.values(): idx=s.index if idx is None else idx.intersection(s.index)
    idx=idx.sort_values()
    Sdf=pd.DataFrame({c:S[c].reindex(idx).ffill() for c in coins})
    Adf=pd.DataFrame({c:A[c].reindex(idx).ffill() for c in coins})
    Vdf=pd.DataFrame({c:V[c].reindex(idx).ffill() for c in coins})
    n=len(idx); cut=int(n*0.6)
    seg=Sdf.iloc[cut:]; brd=Adf.iloc[cut:].mean(axis=1).to_numpy()
    vol=Vdf.iloc[cut:].to_numpy()
    rr=(seg/seg.iloc[0]).pct_change().fillna(0).to_numpy()
    expo=np.concatenate([[1.0],brd[:-1]])
    # equal-weight
    eqw=np.cumprod(1+expo*np.mean(rr,axis=1))
    # inverse-vol weights (lagged 1 row = causal), renormalized each day
    invv=1.0/np.where(vol>0,vol,np.nan)
    invv=np.vstack([np.full(invv.shape[1],np.nan), invv[:-1]])  # lag 1
    w=invv/np.nansum(invv,axis=1,keepdims=True)
    w=np.where(np.isnan(w),1.0/rr.shape[1],w)
    ivp=np.cumprod(1+expo*np.nansum(w*rr,axis=1))
    me,de=eqw[-1]/eqw[0],dd(eqw); mi,di=ivp[-1]/ivp[0],dd(ivp)
    print(f"{label}: equal-wt {me:.3f}x DD{de:.0f}% Cal{cal(me,de):.3f} | "
          f"inv-vol {mi:.3f}x DD{di:.0f}% Cal{cal(mi,di):.3f} | inv-vol better: {cal(mi,di)>cal(me,de)}")
    return {"eq_cal":cal(me,de),"iv_cal":cal(mi,di),"iv_better":bool(cal(mi,di)>cal(me,de))}

r1=run(["BTC","ETH","BNB","SOL","ADA","LINK"],"MAJORS  ")
r2=run(["XRP","LTC","ADA","LINK"],"2nd-set ")
both = r1["iv_better"] and r2["iv_better"]
print(f"\nVERDICT: inv-vol beats equal-weight OOS Calmar in BOTH sets: {both} -> "
      f"{'ACCEPT' if both else 'REJECT (wash/not robust)'}")
json.dump({"majors":r1,"second":r2,"accept":both},open(f"{TMP}/cycle11_result.json","w"),indent=1)
