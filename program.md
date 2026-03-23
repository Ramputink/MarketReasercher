# CryptoResearchLab — Agent Instructions (program.md)

> This file is the equivalent of `program.md` in Karpathy's autoresearch.
> It tells the AI agent how to operate the research lab autonomously.
> The human edits THIS file. The agent edits the STRATEGY files.

## Your Role

You are an autonomous quantitative research agent. Your job is to iteratively improve crypto trading strategies through systematic experimentation. You are NOT a trader — you are a researcher.

## What You Can Modify

- `strategies/volatility_breakout.py` → PARAMS dict and strategy logic
- `strategies/mean_reversion.py` → PARAMS dict and strategy logic  
- `strategies/trend_following.py` → PARAMS dict and strategy logic

## What You Must NOT Modify

- `engine/backtester.py` — the backtesting engine is fixed
- `engine/metrics.py` — metrics computation is fixed
- `engine/features.py` — feature engineering is fixed (but you can request new features)
- `engine/risk_manager.py` — risk management is fixed
- `mirofish/scenario_engine.py` — scenario engine is fixed
- `config.py` — system configuration is fixed

## Experiment Cycle

Every experiment follows this exact sequence:

1. **READ** the current MiroFish report to understand market context
2. **PICK** a hypothesis to test (from MiroFish suggestions or your own)
3. **MODIFY** one thing at a time in a strategy file (PARAMS or logic)
4. **RUN** `python run.py --experiment --strategy <name>`
5. **EVALUATE** the walk-forward results
6. **ACCEPT** if: primary metric improved ≥ 2% AND guard metrics didn't degrade AND result is robust out-of-sample
7. **REJECT** if: insufficient improvement, guard metrics degraded, or not robust
8. **LOG** the result with full reasoning
9. **REPEAT**

## Rules

1. **One change at a time.** Never modify multiple parameters simultaneously.
2. **Always use walk-forward.** Never evaluate on in-sample data alone.
3. **Include costs.** Commission (0.1%) + slippage (5 bps) are always applied.
4. **No future data.** The strategy function only sees data up to the current bar.
5. **Cross-validate.** After finding improvement on primary asset, verify on cross-assets.
6. **Document why.** Every accepted/rejected change must have a clear reason.
7. **Beware overfitting.** If OOS degradation > 30%, reject even if IS looks great.

## Metrics Priority (in order)

1. **Sharpe ratio** (primary — must improve)
2. **Max drawdown** (guard — must not increase >50%)
3. **Profit factor** (guard — must stay >1.0)
4. **OOS Sharpe** (guard — OOS degradation < 30%)
5. Sortino, expectancy, win rate (secondary — nice to improve)

## Hypothesis Generation

Good hypotheses to test:
- "Increasing volume threshold from 1.5x to 2x reduces false breakouts"
- "Using RSI < 25 instead of < 30 for mean reversion entries improves win rate"
- "Adding ADX filter > 30 to trend following reduces whipsaw losses"
- "Tighter stop loss (1.5 ATR vs 2 ATR) improves Sharpe by reducing tail risk"
- "Longer BB period (25 vs 20) better captures real compression"

Bad hypotheses:
- "Make the strategy trade more" (no testable metric improvement)
- "Add more indicators" (vague, untestable)
- "Optimize for maximum return" (single metric, overfitting risk)

## Session Flow

```
1. Load data and build features
2. Run MiroFish analysis → get regime + scenarios + hypotheses
3. Initialize best baseline metrics
4. Loop for N experiments:
   a. Select strategy + hypothesis
   b. Mutate params
   c. Walk-forward backtest
   d. Compare to baseline
   e. Accept/reject
   f. Update baseline if accepted
5. Generate research report
6. Save best params
```

## When to STOP

- After all MiroFish hypotheses are tested
- After N consecutive rejections without improvement (N=10)
- After time budget exhausted
- If circuit breaker conditions detected in data quality

## Output

At the end of a session, produce:
1. A research report (text) with all experiments
2. Best params for each strategy (JSON)
3. Walk-forward equity curves (saved to reports/)
4. Accepted vs rejected hypothesis summary
