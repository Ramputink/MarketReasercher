"""Cycle 8: does the buy-the-dip-to-band entry improve the breadth BMSB portfolio OOS?
Compare dip_entry False vs True on majors (breadth-scaled portfolio), TEST40."""
import sys, os, json, copy, importlib
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP="/Users/matveypro/.claude/jobs/1ca96668/tmp"; COINS=["BTC","ETH","BNB","SOL","ADA","LINK"]

def sleeve_and_band(coin, dip):
    df=pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    df=df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    reg=df["_regime"].to_numpy()
    mod=importlib.import_module("strategies.bmsb_long")
    base=copy.deepcopy(mod.PARAMS); base["dip_entry"]=dip; mod.PARAMS=base
    fn0=mod.bmsb_long_strategy
    bt=BacktestConfig(max_position_pct=1.0); rc=RiskConfig(); rc.sizing_method="fixed_pct"
    def fn(d,i,p):
        r=reg[i] if 0<=i<len(reg) else "unknown"; return fn0(d,i,p,regime=r)
    _t,eq,_m=Backtester(bt,rc).run(df,fn,"bmsb_long")
    ts=pd.to_datetime(df["timestamp"].iloc[-len(eq):].values,unit="ms")
    sl=pd.Series(eq.values,index=ts); sl=sl/sl.iloc[0]
    bt_top=df[["bmsb_sma_real","bmsb_ema_real"]].max(axis=1).values
    ab=pd.Series((df["close"].values>bt_top).astype(float),index=pd.to_datetime(df["timestamp"].values,unit="ms"))
    return sl,ab

def dd(c): c=np.asarray(c,float); pk=np.maximum.accumulate(c); return float(((pk-c)/pk).max()*100)
def cal(m,d): return m/(1+d/100)

def breadth_port(dip):
    S={};A={}
    for c in COINS:
        s,a=sleeve_and_band(c,dip); S[c]=s; A[c]=a
    idx=None
    for s in S.values(): idx=s.index if idx is None else idx.intersection(s.index)
    idx=idx.sort_values()
    Sdf=pd.DataFrame({c:S[c].reindex(idx).ffill() for c in COINS})
    Adf=pd.DataFrame({c:A[c].reindex(idx).ffill() for c in COINS})
    n=len(idx); cut=int(n*0.6)
    seg=Sdf.iloc[cut:]; brd=Adf.iloc[cut:].mean(axis=1).to_numpy()
    segN=seg/seg.iloc[0]; rr=segN.pct_change().fillna(0).to_numpy()
    expo=np.concatenate([[1.0],brd[:-1]])
    eqc=np.cumprod(1+expo*np.mean(rr,axis=1))
    return eqc[-1]/eqc[0], dd(eqc)

for dip in [False, True]:
    m,d=breadth_port(dip)
    print(f"dip_entry={str(dip):<5} TEST40 breadth-portfolio: {m:.3f}x  DD {d:.1f}%  Calmar {cal(m,d):.3f}")
mb,db=breadth_port(False); md,dd_=breadth_port(True)
better = cal(md,dd_) > cal(mb,db)
print(f"\nVERDICT: dip-entry {'IMPROVES' if better else 'does NOT improve'} OOS Calmar "
      f"({cal(md,dd_):.3f} vs {cal(mb,db):.3f})")
json.dump({"flat_mult":mb,"flat_dd":db,"dip_mult":md,"dip_dd":dd_,"dip_better":bool(better)},
          open(f"{TMP}/cycle8_result.json","w"),indent=1)
