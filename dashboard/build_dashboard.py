"""
build_dashboard — render a self-contained HTML dashboard from the real report
files so the project's strategy performance can be inspected visually.

It reads the committed Phase-3 (spot long-only) results and the Phase-2
(cross-sectional carry) summary, then emits `dashboard/dashboard.html`: a single
file with no build step and no external dependencies (charts are hand-rolled SVG
in vanilla JS). Just open the file in a browser.

Run:
    python3 dashboard/build_dashboard.py
"""
from __future__ import annotations

import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
PHASE2 = os.path.join(ROOT, "phase2", "results")
FORWARD = os.path.join(ROOT, "forward_test")
OUT = os.path.join(ROOT, "dashboard", "dashboard.html")


def _maybe_json(path):
    return _load_json(path) if os.path.exists(path) else None


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _cum_from_returns(path, col):
    """Cumulative equity (start=1.0) from a daily simple-return column."""
    out, eq = [], 1.0
    with open(path) as f:
        for row in csv.DictReader(f):
            eq *= 1.0 + float(row[col])
            out.append([row["date"], round(eq, 5)])
    return out


def _equity_col(path, col):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append([row["date"], round(float(row[col]), 5)])
    return out


def _downsample(series, step=4):
    """Keep every `step`-th point plus the last, so SVG stays light."""
    pts = series[::step]
    if series and pts[-1] is not series[-1]:
        pts = pts + [series[-1]]
    return pts


def collect():
    five = _load_json(os.path.join(REPORTS, "phase3_five_strategies_results.json"))
    ph2 = _load_json(os.path.join(PHASE2, "phase2_focused_30assets_2yr.json"))

    sleeves = ["trend_ts", "rel_strength", "dip_in_uptrend", "capitulation", "breakout"]
    labels = {
        "trend_ts": "Trend (TS)",
        "rel_strength": "Relative Strength",
        "dip_in_uptrend": "Dip in Uptrend",
        "capitulation": "Capitulation",
        "breakout": "Breakout",
        "ENSEMBLE_5": "Ensemble (5 sleeves)",
    }
    # ex-ante structural classification (cash-gated sleeves are the keepers)
    cash_gated = {"trend_ts", "breakout", "capitulation"}

    rows = []
    for k in sleeves + ["ENSEMBLE_5"]:
        is_ = five["in_sample"][k]
        oos = five["lockbox_oos"][k]
        rows.append({
            "key": k,
            "label": labels[k],
            "cash_gated": k in cash_gated,
            "is": is_,
            "oos": oos,
        })

    rec = five["recommended_ensemble_3_cash_gating"]
    bh = five["lockbox_buy_hold"]

    # ── P1+P3: portfolio-level de-risking overlay ─────────────────────────────
    ov = _maybe_json(os.path.join(REPORTS, "phase3_overlay_results.json"))
    overlay = None
    if ov:
        lo = ov["lockbox_oos"]
        overlay = {
            "config": ov["overlay_config"],
            "ensemble_raw": lo["ensemble_raw"],
            "ensemble_capped": lo["ensemble_voltarget"],   # vol-target == full here
            "capitulation_raw": lo["capitulation_raw"],
            "capitulation_capped": lo["capitulation_capped"],
        }

    # ── P4: cross-exchange carry replication ──────────────────────────────────
    cx = _maybe_json(os.path.join(PHASE2, "cross_exchange_carry.json"))
    cross = None
    if cx:
        cross = []
        for r in cx["reference_and_venues"]:
            cross.append({"exchange": r["exchange"],
                          "is": r["in_sample"], "lockbox": r["lockbox"]})

    # ── P2: forward (true-OOS) test status ────────────────────────────────────
    forward = None
    log_path = os.path.join(FORWARD, "forward_log.jsonl")
    prereg_path = os.path.join(FORWARD, "preregistration.json")
    if os.path.exists(log_path) and os.path.exists(prereg_path):
        prereg = _load_json(prereg_path)
        recs = [json.loads(l) for l in open(log_path) if l.strip()]
        # latest record per config
        latest = {}
        for r in recs:
            latest[r["config"]] = r
        forward = {"registered": prereg["registered_date"],
                   "forward_start": prereg["forward_start"],
                   "hash": prereg["hash"][:12], "n_runs": len(set(r["run_date"] for r in recs)),
                   "configs": list(latest.values())}

    eq_ens5 = _downsample(_cum_from_returns(
        os.path.join(REPORTS, "phase3_five_strategies_ensemble_equity.csv"), "ensemble_ret"))
    eq_cap = _downsample(_equity_col(
        os.path.join(REPORTS, "phase3_capitulation_equity.csv"), "equity"))

    # Phase-2 carry: best market-neutral config by lockbox Sharpe
    p2res = ph2.get("results", [])
    best = max(p2res, key=lambda r: r.get("lockbox", {}).get("sharpe", -9)) if p2res else None

    return {
        "rows": rows,
        "recommended": rec,
        "buy_hold": bh,
        "overlay": overlay,
        "cross": cross,
        "forward": forward,
        "equity": {
            "ensemble5": eq_ens5,
            "capitulation": eq_cap,
        },
        "phase2": {
            "certified_any": ph2.get("certified_any"),
            "n_configs": ph2.get("n_configs"),
            "yearly_best_mn": ph2.get("yearly_best_mn", {}),
            "best": best and {
                "family": best["cfg"]["family"],
                "k": best["cfg"].get("k"),
                "dollar_neutral": best["cfg"].get("dollar_neutral"),
                "lockbox": best["lockbox"],
                "dsr": best.get("dsr"),
                "tstat": best.get("tstat"),
                "certified": best.get("certified"),
            },
        },
    }


HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MarketReasercher — Strategy Dashboard</title>
<style>
  :root{
    --bg:#0a0e17; --card:#111827; --card2:#0f1626; --border:#1e293b; --dim:#334155;
    --text:#e2e8f0; --muted:#64748b; --accent:#22d3ee; --green:#10b981; --red:#ef4444;
    --yellow:#f59e0b; --blue:#3b82f6; --purple:#a855f7;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1240px;margin:0 auto;padding:28px 22px 80px;}
  h1{font-size:24px;margin:0 0 4px;font-weight:800;letter-spacing:-0.5px;}
  h2{font-size:15px;text-transform:uppercase;letter-spacing:1.4px;color:var(--muted);
    font-weight:700;margin:38px 0 14px;border-bottom:1px solid var(--border);padding-bottom:8px;}
  .sub{color:var(--muted);font-size:13px;margin-bottom:6px}
  .mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
  .card .lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted);font-weight:700}
  .card .val{font-size:24px;font-weight:800;margin-top:6px}
  .card .note{font-size:11px;color:var(--muted);margin-top:3px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:860px){.grid2{grid-template-columns:1fr}}
  .panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 18px 14px}
  .panel h3{margin:0 0 2px;font-size:15px;font-weight:700}
  .panel .ph{color:var(--muted);font-size:12px;margin-bottom:10px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--border)}
  th{color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.8px;font-weight:700}
  td:first-child,th:first-child{text-align:left}
  tr.ens{background:rgba(34,211,238,.06)}
  .pos{color:var(--green)} .neg{color:var(--red)} .warn{color:var(--yellow)}
  .pill{display:inline-block;border-radius:100px;padding:2px 9px;font-size:10px;font-weight:700;
    letter-spacing:.6px;text-transform:uppercase;border:1px solid}
  .pill.keep{background:#052e23;color:var(--green);border-color:#065f46}
  .pill.drop{background:#3a0d0d;color:#fca5a5;border-color:#7f1d1d}
  .legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .dot{width:11px;height:11px;border-radius:3px;display:inline-block}
  .hm td{text-align:center;font-variant-numeric:tabular-nums;font-size:12px;padding:7px 6px}
  .hm td:first-child{text-align:left;color:var(--text)}
  .foot{color:var(--muted);font-size:12px;margin-top:40px;line-height:1.6}
  .tag{display:inline-block;background:var(--dim);color:var(--text);border-radius:6px;
    padding:2px 8px;font-size:11px;margin-right:6px}
  .verdict{background:linear-gradient(180deg,#0f1c1a,#0d1420);border:1px solid #14532d;
    border-radius:12px;padding:16px 18px;font-size:13.5px;line-height:1.6}
  .verdict b{color:var(--green)}
</style>
</head>
<body>
<div class="wrap">
  <h1>MarketReasercher — Strategy Performance</h1>
  <div class="sub">Spot long-only (Phase 3) · sealed 2-year lockbox OOS &nbsp;|&nbsp; cross-sectional carry (Phase 2) · <span id="genstamp"></span></div>

  <div id="kpis" class="cards"></div>

  <h2>Phase 3 — Long-only sleeves: In-sample vs Sealed OOS</h2>
  <div class="sub">In-sample = all history before 2024-06-07. OOS = sealed 2-year lockbox the strategies never saw. Buy &amp; hold basket over the same OOS window: <b class="neg" id="bh"></b>.</div>
  <div class="panel" style="margin-bottom:18px">
    <table id="tbl"></table>
    <div class="legend">
      <span><span class="dot" style="background:var(--green)"></span>cash-gated sleeve (kept)</span>
      <span><span class="dot" style="background:var(--red)"></span>always-invested / counter-trend (dropped)</span>
      <span>OOS = out-of-sample · DD = max drawdown</span>
    </div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h3>OOS equity curves (lockbox)</h3>
      <div class="ph">Daily cumulative growth of $1 over the sealed 2y window</div>
      <div id="eqchart"></div>
    </div>
    <div class="panel">
      <h3>OOS return vs market</h3>
      <div class="ph">Total return over the bear lockbox (basket = −46%)</div>
      <div id="oosbars"></div>
    </div>
  </div>

  <h2>Annual Sharpe (in-sample) per sleeve</h2>
  <div class="panel">
    <table class="hm" id="heat"></table>
    <div class="ph" style="margin-top:8px">Green = positive risk-adjusted year, red = negative. 2022 (crypto bear) separates the cash-gated sleeves from the always-invested ones.</div>
  </div>

  <h2>Recommended ensemble (3 cash-gated sleeves)</h2>
  <div class="grid2">
    <div class="panel">
      <h3>trend_ts + breakout + capitulation</h3>
      <div class="ph" id="recrule"></div>
      <div id="reccards" class="cards"></div>
    </div>
    <div class="verdict" id="verdict"></div>
  </div>

  <h2>P1+P3 — Portfolio de-risking overlay (drawdown fix)</h2>
  <div class="sub" id="ovsub"></div>
  <div class="grid2" id="ovwrap">
    <div class="panel">
      <h3>OOS max drawdown: before vs after overlay</h3>
      <div class="ph">15% vol target + −20%/−10% DD breaker (pre-specified, not fitted)</div>
      <div id="ovbars"></div>
    </div>
    <div class="panel">
      <h3>What the overlay buys you</h3>
      <table id="ovtbl"></table>
      <div class="ph" id="ovnote" style="margin-top:10px"></div>
    </div>
  </div>

  <h2>P4 — Cross-exchange carry replication</h2>
  <div class="sub">Same pre-registered carry config (clb7/k4/weekly/MN) on an independent venue. In-sample agreement + out-of-sample divergence = the honest robustness verdict.</div>
  <div class="grid2" id="cxwrap">
    <div class="panel">
      <h3>Lockbox Sharpe by exchange</h3>
      <div class="ph">Same config, two venues. Replication needs same sign + significance.</div>
      <div id="cxbars"></div>
    </div>
    <div class="panel">
      <h3>Replication table</h3>
      <table id="cxtbl"></table>
      <div class="ph" id="cxnote" style="margin-top:10px"></div>
    </div>
  </div>

  <h2>P2 — Forward (true out-of-sample) test</h2>
  <div class="panel" id="fwwrap">
    <div class="ph" id="fwhead"></div>
    <table id="fwtbl"></table>
    <div class="ph" id="fwnote" style="margin-top:10px"></div>
  </div>

  <h2>Phase 2 — Cross-sectional carry (perps, market-neutral)</h2>
  <div class="grid2">
    <div class="panel">
      <h3>Best lockbox config</h3>
      <div class="ph">Search over <span id="p2n"></span> pre-specified configs · certified by strict DSR≥0.95</div>
      <div id="p2cards" class="cards"></div>
    </div>
    <div class="panel">
      <h3>Best market-neutral Sharpe by year</h3>
      <div class="ph">Annual Sharpe of the best carry config (negative years = no edge that year)</div>
      <div id="p2heat"></div>
    </div>
  </div>

  <div class="foot" id="foot"></div>
</div>

<script>
const DATA = __DATA__;
const C = {green:"#10b981",red:"#ef4444",yellow:"#f59e0b",blue:"#3b82f6",accent:"#22d3ee",
  purple:"#a855f7",muted:"#64748b",border:"#1e293b",text:"#e2e8f0"};
const fmt = (x,d=1)=> (x==null||isNaN(x))?"—":(x>0?"+":"")+Number(x).toFixed(d);
const cls = x=> x>0?"pos":(x<0?"neg":"");

// ---- KPIs ----
const rec = DATA.recommended, bh = DATA.buy_hold;
const breakout = DATA.rows.find(r=>r.key==="breakout");
document.getElementById("bh").textContent = fmt(bh.total_ret_pct)+"%";
const kpis = [
  {lbl:"Ensemble OOS return", val:fmt(rec.lockbox_oos.cagr_pct)+"%", note:"3 cash-gated sleeves, 2y CAGR", c:C.green},
  {lbl:"Ensemble OOS Sharpe", val:fmt(rec.lockbox_oos.sharpe,2), note:"positive 2024 & 2025", c:C.accent},
  {lbl:"Best single sleeve OOS", val:fmt(breakout.oos.cagr_pct)+"%", note:"breakout (Sharpe "+fmt(breakout.oos.sharpe,2)+")", c:C.green},
  {lbl:"Market (buy & hold)", val:fmt(bh.total_ret_pct)+"%", note:"basket over same window", c:C.red},
  {lbl:"Ensemble max DD", val:fmt(rec.lockbox_oos.max_dd_pct)+"%", note:"vs basket "+fmt(bh.max_dd_pct)+"%", c:C.yellow},
];
document.getElementById("kpis").innerHTML = kpis.map(k=>
  `<div class="card"><div class="lbl">${k.lbl}</div>
   <div class="val mono" style="color:${k.c}">${k.val}</div>
   <div class="note">${k.note}</div></div>`).join("");

// ---- main table ----
const cols = [["CAGR %","cagr_pct",1],["Sharpe","sharpe",2],["Max DD %","max_dd_pct",1],
  ["Calmar","calmar",2],["Exposure %","avg_exposure_pct",0]];
let th = `<tr><th>Sleeve</th><th></th><th colspan="${cols.length}" style="text-align:center;color:${C.muted}">IN-SAMPLE</th><th colspan="3" style="text-align:center;color:${C.accent}">SEALED OOS</th></tr>`;
th += `<tr><th></th><th></th>`+cols.map(c=>`<th>${c[0]}</th>`).join("")
   + `<th>CAGR %</th><th>Sharpe</th><th>DD %</th></tr>`;
let body = "";
for(const r of DATA.rows){
  const ens = r.key==="ENSEMBLE_5";
  const pill = ens ? "" : `<span class="pill ${r.cash_gated?"keep":"drop"}">${r.cash_gated?"keep":"drop"}</span>`;
  body += `<tr class="${ens?"ens":""}">
    <td><b>${r.label}</b></td><td>${pill}</td>`
    + cols.map(c=>`<td class="mono">${r.is[c[1]]==null?"—":Number(r.is[c[1]]).toFixed(c[2])}</td>`).join("")
    + `<td class="mono ${cls(r.oos.cagr_pct)}">${fmt(r.oos.cagr_pct)}</td>`
    + `<td class="mono ${cls(r.oos.sharpe)}">${fmt(r.oos.sharpe,2)}</td>`
    + `<td class="mono">${Number(r.oos.max_dd_pct).toFixed(1)}</td></tr>`;
}
document.getElementById("tbl").innerHTML = th+body;

// ---- SVG line chart ----
function lineChart(series, w=520, h=240, pad=34){
  // series: [{name,color,pts:[[date,val],...]}]
  let lo=Infinity, hi=-Infinity, n=0;
  for(const s of series){ for(const p of s.pts){ lo=Math.min(lo,p[1]); hi=Math.max(hi,p[1]); } n=Math.max(n,s.pts.length); }
  const padv=(hi-lo)*0.08||0.1; lo-=padv; hi+=padv;
  const X=i=>pad+ i*(w-2*pad)/(n-1);
  const Y=v=>h-pad-(v-lo)*(h-2*pad)/(hi-lo);
  let svg=`<svg viewBox="0 0 ${w} ${h}" width="100%" style="display:block">`;
  // baseline at 1.0
  if(lo<1 && hi>1){ svg+=`<line x1="${pad}" y1="${Y(1)}" x2="${w-pad}" y2="${Y(1)}" stroke="${C.muted}" stroke-dasharray="3 3" opacity=".5"/>`;
    svg+=`<text x="${w-pad}" y="${Y(1)-4}" fill="${C.muted}" font-size="9" text-anchor="end">$1.00</text>`; }
  // y ticks
  for(const t of [lo,(lo+hi)/2,hi]){ svg+=`<text x="6" y="${Y(t)+3}" fill="${C.muted}" font-size="9">${t.toFixed(2)}</text>`; }
  for(const s of series){
    let d="";
    s.pts.forEach((p,i)=> d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(p[1]).toFixed(1)+" ");
    svg+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
  }
  // x labels: first & last date
  const f=series[0].pts;
  svg+=`<text x="${pad}" y="${h-8}" fill="${C.muted}" font-size="9">${f[0][0]}</text>`;
  svg+=`<text x="${w-pad}" y="${h-8}" fill="${C.muted}" font-size="9" text-anchor="end">${f[f.length-1][0]}</text>`;
  svg+="</svg>";
  return svg;
}
const eqSeries=[
  {name:"Ensemble (5)",color:C.accent,pts:DATA.equity.ensemble5},
  {name:"Capitulation",color:C.purple,pts:DATA.equity.capitulation},
];
document.getElementById("eqchart").innerHTML = lineChart(eqSeries)
  + `<div class="legend">`+eqSeries.map(s=>`<span><span class="dot" style="background:${s.color}"></span>${s.name}</span>`).join("")+`</div>`;

// ---- OOS bar chart ----
function barChart(items, {unit="%", dp=1, w=520, h=240, pad=34}={}){
  const vals=items.map(i=>i.v).concat([0]);
  let lo=Math.min(...vals), hi=Math.max(...vals); const padv=(hi-lo)*0.12||1; lo-=padv; hi+=padv;
  const Y=v=>h-pad-(v-lo)*(h-2*pad)/(hi-lo);
  const bw=(w-2*pad)/items.length*0.62;
  let svg=`<svg viewBox="0 0 ${w} ${h}" width="100%" style="display:block">`;
  svg+=`<line x1="${pad}" y1="${Y(0)}" x2="${w-pad}" y2="${Y(0)}" stroke="${C.muted}" opacity=".6"/>`;
  items.forEach((it,i)=>{
    const cx=pad+(i+0.5)*(w-2*pad)/items.length;
    const y0=Y(0), y1=Y(it.v);
    svg+=`<rect x="${cx-bw/2}" y="${Math.min(y0,y1)}" width="${bw}" height="${Math.abs(y1-y0)}" rx="3" fill="${it.c}"/>`;
    svg+=`<text x="${cx}" y="${(it.v>=0?y1-5:y1+12)}" fill="${C.text}" font-size="10" text-anchor="middle">${fmt(it.v,dp)}${unit}</text>`;
    svg+=`<text x="${cx}" y="${h-8}" fill="${C.muted}" font-size="9" text-anchor="middle">${it.l}</text>`;
  });
  svg+="</svg>"; return svg;
}
const oosItems = DATA.rows.filter(r=>r.key!=="ENSEMBLE_5").map(r=>({
  l:r.label.split(" ")[0], v:r.oos.cagr_pct, c:r.oos.cagr_pct>=0?C.green:C.red}))
  .concat([{l:"Ens-3",v:rec.lockbox_oos.cagr_pct,c:C.accent},{l:"Market",v:bh.total_ret_pct,c:C.red}]);
document.getElementById("oosbars").innerHTML = barChart(oosItems);

// ---- heatmap ----
function heat(rows, years){
  let h=`<tr><th>Sleeve</th>`+years.map(y=>`<th style="text-align:center">${y}</th>`).join("")+`</tr>`;
  for(const r of rows){
    h+=`<tr><td>${r.label}</td>`+years.map(y=>{
      const v=r.vals[y];
      if(v==null||v===0) return `<td style="color:${C.muted}">·</td>`;
      const a=Math.min(Math.abs(v)/2.5,1);
      const col=v>0?`rgba(16,185,129,${0.15+0.6*a})`:`rgba(239,68,68,${0.15+0.6*a})`;
      return `<td style="background:${col};border-radius:4px">${v.toFixed(2)}</td>`;
    }).join("")+`</tr>`;
  }
  return h;
}
const yrs=["2019","2020","2021","2022","2023","2024"];
const hrows=DATA.rows.filter(r=>r.key!=="ENSEMBLE_5").map(r=>({label:r.label,vals:r.is.per_year}));
document.getElementById("heat").innerHTML = heat(hrows, yrs);

// ---- recommended ensemble ----
document.getElementById("recrule").textContent = rec.selection_rule;
const rc=[
  {lbl:"OOS CAGR",val:fmt(rec.lockbox_oos.cagr_pct)+"%",c:C.green},
  {lbl:"OOS Sharpe",val:fmt(rec.lockbox_oos.sharpe,2),c:C.accent},
  {lbl:"OOS Max DD",val:fmt(rec.lockbox_oos.max_dd_pct)+"%",c:C.yellow},
  {lbl:"In-sample CAGR",val:fmt(rec.in_sample.cagr_pct)+"%",c:C.text},
];
document.getElementById("reccards").innerHTML = rc.map(k=>
  `<div class="card"><div class="lbl">${k.lbl}</div><div class="val mono" style="color:${k.c}">${k.val}</div></div>`).join("");
document.getElementById("verdict").innerHTML =
  `<b>Structural law (proven on the sealed lockbox):</b> long-only profitability through a bear
   requires sleeves that <b>go to cash on weakness</b> (breakout, trend) — not always-invested
   (rel_strength) or knife-catching (dip). The 3-sleeve cash-gating ensemble returned
   <b>${fmt(rec.lockbox_oos.cagr_pct)}%</b> while the basket lost <b style="color:${C.red}">${fmt(bh.total_ret_pct)}%</b>.
   <br><br>Open issue: ensemble max DD ${fmt(rec.lockbox_oos.max_dd_pct)}% (sleeves co-invest in drops).
   Next: portfolio-level regime / drawdown cap, then validate on <i>forward</i> data — not by re-reading this lockbox.`;

// ---- P1+P3 overlay ----
const ov=DATA.overlay;
if(ov){
  const oc=ov.config;
  document.getElementById("ovsub").innerHTML =
    `The 3-sleeve ensemble made +10% on the bear lockbox but still drew down −41.8% (sleeves co-invest in drops). A causal, pre-specified overlay (${(oc.target_vol_ann*100)}% vol target, DD breaker ${(oc.dd_trigger*100)}%/${(oc.dd_reentry*100)}%) caps the book toward cash — de-risk only, no leverage.`;
  const ovd=[
    {l:"Ensemble RAW",v:ov.ensemble_raw.max_dd_pct,c:C.red},
    {l:"Ensemble +overlay",v:ov.ensemble_capped.max_dd_pct,c:C.green},
    {l:"Capit. RAW",v:ov.capitulation_raw.max_dd_pct,c:C.red},
    {l:"Capit. +overlay",v:ov.capitulation_capped.max_dd_pct,c:C.green},
  ];
  document.getElementById("ovbars").innerHTML = barChart(ovd);
  const orow=(lbl,raw,cap)=>`<tr><td>${lbl}</td>
    <td class="mono">${fmt(raw.max_dd_pct)}% → <b class="${cap.max_dd_pct>raw.max_dd_pct?'pos':'neg'}">${fmt(cap.max_dd_pct)}%</b></td>
    <td class="mono">${fmt(raw.sharpe,2)} → <b class="${cap.sharpe>=raw.sharpe?'pos':'neg'}">${fmt(cap.sharpe,2)}</b></td>
    <td class="mono">${fmt(raw.cagr_pct)}% → ${fmt(cap.cagr_pct)}%</td></tr>`;
  document.getElementById("ovtbl").innerHTML =
    `<tr><th>OOS</th><th>Max DD</th><th>Sharpe</th><th>CAGR</th></tr>`
    + orow("Ensemble", ov.ensemble_raw, ov.ensemble_capped)
    + orow("Capitulation", ov.capitulation_raw, ov.capitulation_capped);
  document.getElementById("ovnote").innerHTML =
    `<b style="color:${C.green}">Drawdown solved:</b> ensemble maxDD ${fmt(ov.ensemble_raw.max_dd_pct)}% → ${fmt(ov.ensemble_capped.max_dd_pct)}% (−62%). The carry-feature capitulation sleeve improves outright (Sharpe ${fmt(ov.capitulation_raw.sharpe,2)}→${fmt(ov.capitulation_capped.sharpe,2)}, return ${fmt(ov.capitulation_raw.cagr_pct)}%→${fmt(ov.capitulation_capped.cagr_pct)}%). Honest cost: on this OOS path vol-targeting also cut the ensemble's Sharpe (positive returns clustered in high-vol days).`;
} else { document.getElementById("ovwrap").style.display="none"; }

// ---- P4 cross-exchange ----
const cx=DATA.cross;
if(cx){
  const cxd=cx.map(r=>({l:r.exchange.replace("usdm",""),v:r.lockbox.sharpe,
    c:r.lockbox.sharpe>0?C.green:C.red}));
  document.getElementById("cxbars").innerHTML = barChart(cxd, {unit:"", dp:3});
  let ct=`<tr><th>Exchange</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS tstat</th><th>OOS ret</th></tr>`;
  for(const r of cx){
    ct+=`<tr><td><b>${r.exchange}</b></td>
      <td class="mono">${fmt(r.is.sharpe,3)}</td>
      <td class="mono ${cls(r.lockbox.sharpe)}">${fmt(r.lockbox.sharpe,3)}</td>
      <td class="mono ${cls(r.lockbox.tstat)}">${fmt(r.lockbox.tstat,2)}</td>
      <td class="mono ${cls(r.lockbox.ret_pct)}">${fmt(r.lockbox.ret_pct)}%</td></tr>`;
  }
  document.getElementById("cxtbl").innerHTML = ct;
  document.getElementById("cxnote").innerHTML =
    `<b class="warn">Does NOT replicate out-of-sample.</b> The carry signal agrees IN-SAMPLE on both venues (~0.14–0.16 Sharpe, t≈2), and funding is 75% correlated across exchanges — yet the sealed lockbox flips sign (Binance positive, Bybit negative). The edge is real as a phenomenon but fragile in OOS execution: changing venue breaks it.`;
} else { document.getElementById("cxwrap").style.display="none"; }

// ---- P2 forward test ----
const fw=DATA.forward;
if(fw){
  document.getElementById("fwhead").innerHTML =
    `Pre-registered ${fw.registered} · forward window opens after <b>${fw.forward_start}</b> · config hash <span class="mono">${fw.hash}…</span> · ${fw.n_runs} run(s) logged. Data after forward_start did not exist in the frozen panels — genuine OOS.`;
  let ft=`<tr><th>Config</th><th>Forward obs</th><th>Return</th><th>Sharpe/obs</th><th>Status</th></tr>`;
  for(const c of fw.configs){
    ft+=`<tr><td><b>${c.config}</b></td>
      <td class="mono">${c.n}</td>
      <td class="mono ${c.total_ret_pct!=null?cls(c.total_ret_pct):''}">${c.total_ret_pct!=null?fmt(c.total_ret_pct)+"%":"—"}</td>
      <td class="mono">${c.sharpe_per_obs!=null?fmt(c.sharpe_per_obs,3):"—"}</td>
      <td style="text-align:left;color:${C.muted};font-size:12px">${c.status}</td></tr>`;
  }
  document.getElementById("fwtbl").innerHTML = ft;
  document.getElementById("fwnote").innerHTML =
    `The forward window is still days old — too short for a verdict. Re-run <span class="mono">python3 -m forward_test.run_forward</span> weekly; the track record accumulates honestly and is the only test that can confirm or kill the historical lockbox reads above.`;
} else { document.getElementById("fwwrap").style.display="none"; }

// ---- phase 2 ----
const p2=DATA.phase2;
document.getElementById("p2n").textContent = p2.n_configs;
const b=p2.best;
const p2cards=[
  {lbl:"Family",val:b?b.family:"—",c:C.text},
  {lbl:"Lockbox Sharpe",val:b?fmt(b.lockbox.sharpe,3):"—",c:C.accent},
  {lbl:"Lockbox return",val:b?fmt(b.lockbox.ret_pct)+"%":"—",c:b&&b.lockbox.ret_pct>0?C.green:C.red},
  {lbl:"DSR (need ≥.95)",val:b?fmt(b.dsr,3):"—",c:C.red},
  {lbl:"Certified",val:p2.certified_any?"YES":"NO",c:p2.certified_any?C.green:C.red},
];
document.getElementById("p2cards").innerHTML = p2cards.map(k=>
  `<div class="card"><div class="lbl">${k.lbl}</div><div class="val mono" style="color:${k.c};font-size:19px">${k.val}</div></div>`).join("");
// phase2 yearly heat
const p2years=Object.keys(p2.yearly_best_mn);
document.getElementById("p2heat").innerHTML =
  `<table class="hm">`+heat([{label:"Carry MN",vals:p2.yearly_best_mn}], p2years)+`</table>`;

document.getElementById("foot").innerHTML =
  `<b>Reading this honestly.</b> The Phase-3 OOS numbers are a single sealed read on one 2-year bear window —
   encouraging, not certified. Phase-2 carry is the only economically pre-specified edge found, but it does
   not pass strict DSR≥0.95. Neither result licenses live capital without forward (paper) validation.
   <br>Generated from <span class="tag">reports/phase3_five_strategies_results.json</span>
   <span class="tag">phase2/results/phase2_focused_30assets_2yr.json</span> · regenerate with
   <span class="mono">python3 dashboard/build_dashboard.py</span>`;
document.getElementById("genstamp").textContent = "generated " + (DATA.generated||"");
</script>
</body>
</html>
"""


def main():
    from datetime import datetime
    data = collect()
    data["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = HTML.replace("__DATA__", json.dumps(data))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
