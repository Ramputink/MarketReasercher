"""Cycle 2 (Part A): OOS parameter search for the two intraday long-only strategies
across coins. Train-search params on first 60%, measure frozen params on unseen 40%.
L=1 only (no leverage claims on a thin long-only edge). Honest verdict: does ANY
(strategy, coin) clear 1.0 OOS robustly?"""
import sys, os, json
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, "tools")
import numpy as np
import honest_eval as HE

COINS = ["XRP", "BTC", "ETH", "SOL", "DOGE", "BNB", "ADA", "LINK", "AVAX", "LTC"]
STRATS = ["mr_vwap_reversion", "vol_expansion_long"]
results = {}
for strat in STRATS:
    print(f"\n=== {strat} (OOS test multiple @ L=1, train-searched params) ===")
    print(f"{'coin':<6}{'train':>9}{'TEST_oos':>10}{'trades':>8}")
    rows = []
    for coin in COINS:
        ho = HE.evaluate_holdout(strat, coin, train_frac=0.6, leverages=(1,),
                                 model_liquidation=True, n_search=30)
        tb = ho["TRAIN"]["best"]; eb = ho["TEST(unseen)"]["best"]
        tr = tb["mult"] if tb else 0.0
        te = eb["mult"] if eb else 0.0
        nt = eb["n_trades"] if eb else 0
        rows.append({"coin": coin, "train": tr, "test": te, "trades": nt})
        print(f"{coin:<6}{tr:>9.3f}{te:>10.3f}{nt:>8}")
    tests = [r["test"] for r in rows]
    wins = sum(1 for t in tests if t > 1.02)  # >2% to clear costs/noise
    print(f"  -> OOS mean {np.mean(tests):.3f}x | coins clearing 1.02x: {wins}/{len(rows)} "
          f"| median {np.median(tests):.3f}x")
    results[strat] = {"rows": rows, "oos_mean": float(np.mean(tests)),
                      "oos_median": float(np.median(tests)), "wins": wins, "n": len(rows)}
json.dump(results, open("/Users/matveypro/.claude/jobs/1ca96668/tmp/partA_search_result.json", "w"), indent=1)
print("\nVERDICT:", "edge found" if any(r["wins"] >= len(r["rows"]) * 0.5 for r in results.values())
      else "NO robust long-only intraday edge (most coins <1.0 OOS)")
