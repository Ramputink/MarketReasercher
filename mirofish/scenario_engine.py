"""
CryptoResearchLab — MiroFish Scenario Engine
Multi-agent simulation for market regime classification, scenario generation,
and catalyst detection. Inspired by MiroFish's swarm intelligence approach.

This is NOT a trading signal generator — it produces context and hypotheses
that feed into autoresearch for quantitative validation.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import MiroFishConfig, MarketRegime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# AGENT ARCHETYPES
# ═══════════════════════════════════════════════════════════════

@dataclass
class MarketAgent:
    """A simulated market participant with memory and behavior."""
    archetype: str
    bias: str = "neutral"          # "bullish", "bearish", "neutral"
    confidence: float = 0.5         # 0-1
    time_horizon: str = "medium"    # "short", "medium", "long"
    memory: list = field(default_factory=list)
    last_action: str = "hold"

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "bias": self.bias,
            "confidence": self.confidence,
            "time_horizon": self.time_horizon,
            "last_action": self.last_action,
        }


AGENT_TEMPLATES = {
    "whale_accumulator": MarketAgent(
        archetype="whale_accumulator",
        time_horizon="long",
        bias="neutral",
    ),
    "momentum_trader": MarketAgent(
        archetype="momentum_trader",
        time_horizon="short",
        bias="neutral",
    ),
    "mean_reversion_trader": MarketAgent(
        archetype="mean_reversion_trader",
        time_horizon="short",
        bias="neutral",
    ),
    "news_reactive_trader": MarketAgent(
        archetype="news_reactive_trader",
        time_horizon="short",
        bias="neutral",
    ),
    "market_maker": MarketAgent(
        archetype="market_maker",
        time_horizon="short",
        bias="neutral",
    ),
    "retail_fomo": MarketAgent(
        archetype="retail_fomo",
        time_horizon="short",
        bias="neutral",
    ),
    "institutional_systematic": MarketAgent(
        archetype="institutional_systematic",
        time_horizon="medium",
        bias="neutral",
    ),
}


# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER (QUANTITATIVE)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RegimeClassification:
    """Market regime classification result."""
    regime: MarketRegime
    confidence: float
    supporting_signals: list = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "signals": self.supporting_signals,
            "timestamp": self.timestamp,
        }


def classify_regime_quantitative(df: pd.DataFrame, bar_idx: int = -1) -> RegimeClassification:
    """
    Classify market regime using purely quantitative features.
    No LLM needed — this is the deterministic backbone.

    Regime logic:
    - BREAKOUT: Low BB bandwidth + price near band edges + high volume
    - TREND: High ADX + consistent direction + BMSB alignment
    - MEAN_REVERSION: High BB bandwidth + oscillating price + extreme z-score
    - LATERAL: Low ADX + medium bandwidth + range-bound
    - EXOGENOUS_SHOCK: Sudden vol spike + extreme returns
    """
    if len(df) < 50:
        return RegimeClassification(MarketRegime.UNKNOWN, 0.0)

    current = df.iloc[bar_idx]
    signals = []
    scores = {r: 0.0 for r in MarketRegime}

    # ─── Volatility assessment ───
    bb_bw = current.get("bb_bandwidth", None)
    bb_bw_pct = current.get("bb_bw_percentile", 50)
    vol_ratio = current.get("vol_ratio", 1.0)
    realized_vol = current.get("realized_vol_5", None)

    if bb_bw_pct is not None and not pd.isna(bb_bw_pct):
        if bb_bw_pct < 15:
            scores[MarketRegime.BREAKOUT] += 0.3
            signals.append(f"BB_compression: {bb_bw_pct:.0f}th pctile")
        elif bb_bw_pct > 80:
            scores[MarketRegime.MEAN_REVERSION] += 0.2
            signals.append(f"BB_expansion: {bb_bw_pct:.0f}th pctile")

    # Sudden vol spike → shock
    if vol_ratio is not None and not pd.isna(vol_ratio):
        if vol_ratio > 2.5:
            scores[MarketRegime.EXOGENOUS_SHOCK] += 0.4
            signals.append(f"Vol_spike: ratio={vol_ratio:.1f}")

    # ─── Trend assessment ───
    adx_val = current.get("adx_14", 0)
    bmsb_bullish = current.get("bmsb_bullish", None)

    if adx_val is not None and not pd.isna(adx_val):
        if adx_val > 30:
            scores[MarketRegime.TREND] += 0.35
            signals.append(f"Strong_trend: ADX={adx_val:.0f}")
        elif adx_val < 20:
            scores[MarketRegime.LATERAL] += 0.3
            signals.append(f"Weak_trend: ADX={adx_val:.0f}")

    # ─── Return extremity ───
    ret_zscore = current.get("ret_zscore_20", 0)
    if ret_zscore is not None and not pd.isna(ret_zscore):
        if abs(ret_zscore) > 2.5:
            scores[MarketRegime.MEAN_REVERSION] += 0.25
            scores[MarketRegime.EXOGENOUS_SHOCK] += 0.15
            signals.append(f"Extreme_return: zscore={ret_zscore:.1f}")
        elif abs(ret_zscore) < 0.5:
            scores[MarketRegime.LATERAL] += 0.15

    # ─── Volume profile ───
    vol_zscore = current.get("volume_zscore", 0)
    if vol_zscore is not None and not pd.isna(vol_zscore):
        if vol_zscore > 2:
            scores[MarketRegime.BREAKOUT] += 0.2
            signals.append(f"Volume_surge: zscore={vol_zscore:.1f}")

    # ─── Range assessment ───
    dist_high = current.get("dist_from_high_20", 0)
    dist_low = current.get("dist_from_low_20", 0)
    if dist_high is not None and dist_low is not None:
        if not pd.isna(dist_high) and not pd.isna(dist_low):
            range_span = abs(dist_high) + abs(dist_low)
            if range_span < 0.02:  # Very tight range
                scores[MarketRegime.BREAKOUT] += 0.15
                scores[MarketRegime.LATERAL] += 0.15
            elif range_span > 0.1:  # Wide range
                scores[MarketRegime.MEAN_REVERSION] += 0.1

    # ─── Select winning regime ───
    best_regime = max(scores, key=scores.get)
    best_score = scores[best_regime]
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0

    return RegimeClassification(
        regime=best_regime,
        confidence=confidence,
        supporting_signals=signals,
        timestamp=str(datetime.now(timezone.utc)),
    )


# ═══════════════════════════════════════════════════════════════
# SCENARIO GENERATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """A possible market scenario with probability."""
    name: str
    description: str
    probability: float  # 0-1
    regime: MarketRegime
    expected_direction: str  # "up", "down", "flat"
    expected_magnitude_pct: float
    catalysts: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    strategy_implications: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "probability": round(self.probability, 2),
            "regime": self.regime.value,
            "direction": self.expected_direction,
            "magnitude_%": self.expected_magnitude_pct,
            "catalysts": self.catalysts,
            "risks": self.risks,
            "strategy_implications": self.strategy_implications,
        }


def generate_scenarios_quantitative(
    df: pd.DataFrame,
    regime: RegimeClassification,
    bar_idx: int = -1,
) -> list[Scenario]:
    """
    Generate market scenarios based on current regime and features.
    Deterministic version — no LLM required.
    """
    current = df.iloc[bar_idx]
    close = float(current.get("close", 0))
    atr_val = float(current.get("atr_14", close * 0.02))
    scenarios = []

    atr_pct = (atr_val / close * 100) if close > 0 else 2.0

    if regime.regime == MarketRegime.BREAKOUT:
        scenarios.append(Scenario(
            name="Upside Breakout",
            description="Volatility compression resolves to the upside with volume confirmation",
            probability=0.35,
            regime=MarketRegime.BREAKOUT,
            expected_direction="up",
            expected_magnitude_pct=atr_pct * 2,
            catalysts=["Volume surge", "Order flow imbalance", "Compression exhaustion"],
            risks=["False breakout / head fake", "Lack of follow-through volume"],
            strategy_implications={
                "volatility_breakout": "ACTIVE — primary setup",
                "mean_reversion": "AVOID — breakout environment",
                "trend_following": "STANDBY — wait for trend confirmation",
            },
        ))
        scenarios.append(Scenario(
            name="Downside Breakout",
            description="Compression resolves to the downside",
            probability=0.30,
            regime=MarketRegime.BREAKOUT,
            expected_direction="down",
            expected_magnitude_pct=atr_pct * 2,
            catalysts=["Liquidation cascade", "Support breach"],
            risks=["False breakdown", "Immediate buy-back"],
            strategy_implications={
                "volatility_breakout": "ACTIVE — short setup",
                "mean_reversion": "AVOID",
                "trend_following": "STANDBY",
            },
        ))
        scenarios.append(Scenario(
            name="False Breakout / Whipsaw",
            description="Price tests extremes but fails to sustain",
            probability=0.35,
            regime=MarketRegime.LATERAL,
            expected_direction="flat",
            expected_magnitude_pct=atr_pct,
            catalysts=["Low conviction", "Thin orderbook"],
            risks=["Stopped out on both sides"],
            strategy_implications={
                "volatility_breakout": "CAUTION — false signal risk",
                "mean_reversion": "POSSIBLE — after false breakout",
                "trend_following": "AVOID",
            },
        ))

    elif regime.regime == MarketRegime.TREND:
        direction = "up" if current.get("bmsb_bullish", True) else "down"
        scenarios.append(Scenario(
            name=f"Trend Continuation ({direction})",
            description=f"Strong {'up' if direction == 'up' else 'down'}trend continues with pullback entries",
            probability=0.50,
            regime=MarketRegime.TREND,
            expected_direction=direction,
            expected_magnitude_pct=atr_pct * 3,
            catalysts=["Momentum", "BMSB support", "ADX expansion"],
            risks=["Trend exhaustion", "Regime change"],
            strategy_implications={
                "volatility_breakout": "STANDBY",
                "mean_reversion": "AVOID — trend override",
                "trend_following": "ACTIVE — primary setup",
            },
        ))
        scenarios.append(Scenario(
            name="Trend Reversal",
            description="Current trend exhausts and reverses",
            probability=0.20,
            regime=MarketRegime.MEAN_REVERSION,
            expected_direction="down" if direction == "up" else "up",
            expected_magnitude_pct=atr_pct * 2,
            catalysts=["Divergence", "Volume decline", "Exhaustion candles"],
            risks=["Premature reversal call"],
            strategy_implications={
                "volatility_breakout": "STANDBY",
                "mean_reversion": "POSSIBLE — after confirmed exhaustion",
                "trend_following": "EXIT — stop and reassess",
            },
        ))

    elif regime.regime == MarketRegime.MEAN_REVERSION:
        scenarios.append(Scenario(
            name="Reversion to Mean",
            description="Overextended price reverts to equilibrium",
            probability=0.45,
            regime=MarketRegime.MEAN_REVERSION,
            expected_direction="down" if float(current.get("ret_zscore_20", 0)) > 0 else "up",
            expected_magnitude_pct=atr_pct * 1.5,
            catalysts=["Exhaustion", "Mean attraction", "Profit taking"],
            risks=["Further extension", "New catalyst extends move"],
            strategy_implications={
                "volatility_breakout": "AVOID",
                "mean_reversion": "ACTIVE — primary setup",
                "trend_following": "AVOID — choppy environment",
            },
        ))

    elif regime.regime == MarketRegime.LATERAL:
        scenarios.append(Scenario(
            name="Range Continuation",
            description="Price continues oscillating within established range",
            probability=0.50,
            regime=MarketRegime.LATERAL,
            expected_direction="flat",
            expected_magnitude_pct=atr_pct,
            catalysts=["Low volume", "No catalyst", "Market indecision"],
            risks=["Sudden breakout from range"],
            strategy_implications={
                "volatility_breakout": "PREPARE — compression building",
                "mean_reversion": "POSSIBLE — range extremes",
                "trend_following": "AVOID — no trend",
            },
        ))

    elif regime.regime == MarketRegime.EXOGENOUS_SHOCK:
        scenarios.append(Scenario(
            name="Shock Absorption",
            description="Market absorbs shock and stabilizes",
            probability=0.40,
            regime=MarketRegime.MEAN_REVERSION,
            expected_direction="reversal",
            expected_magnitude_pct=atr_pct * 2,
            catalysts=["Mean reversion", "Panic exhaustion"],
            risks=["Secondary shock", "Cascade effect"],
            strategy_implications={
                "volatility_breakout": "AVOID — too volatile",
                "mean_reversion": "CAUTION — wait for stabilization",
                "trend_following": "AVOID — regime unstable",
            },
        ))

    return scenarios


# ═══════════════════════════════════════════════════════════════
# MIROFISH CONTEXT REPORT
# ═══════════════════════════════════════════════════════════════

@dataclass
class MiroFishReport:
    """Complete context report from MiroFish analysis."""
    timestamp: str
    symbol: str
    regime: RegimeClassification = field(default_factory=lambda: RegimeClassification(MarketRegime.UNKNOWN, 0.0))
    scenarios: list = field(default_factory=list)
    agent_consensus: dict = field(default_factory=dict)
    recommended_strategies: list = field(default_factory=list)
    avoid_strategies: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    hypotheses_for_autoresearch: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "regime": self.regime.to_dict(),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "agent_consensus": self.agent_consensus,
            "recommended_strategies": self.recommended_strategies,
            "avoid_strategies": self.avoid_strategies,
            "risk_flags": self.risk_flags,
            "hypotheses": self.hypotheses_for_autoresearch,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ═══════════════════════════════════════════════════════════════
# MULTI-AGENT SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_agent_reactions(
    df: pd.DataFrame,
    regime: RegimeClassification,
    bar_idx: int = -1,
) -> dict:
    """
    Simulate how different market participant archetypes would react
    to the current market state. Returns consensus summary.
    """
    current = df.iloc[bar_idx]
    agents = {}

    close = float(current.get("close", 0))
    ret_1 = float(current.get("ret_1", 0)) if not pd.isna(current.get("ret_1", np.nan)) else 0
    vol_ratio = float(current.get("volume_ratio", 1)) if not pd.isna(current.get("volume_ratio", np.nan)) else 1
    rsi_val = float(current.get("rsi_14", 50)) if not pd.isna(current.get("rsi_14", np.nan)) else 50

    # Whale: accumulates on dips, distributes on spikes
    whale = MarketAgent(archetype="whale_accumulator", time_horizon="long")
    if ret_1 < -0.02:
        whale.bias = "bullish"
        whale.last_action = "accumulate"
        whale.confidence = min(abs(ret_1) * 10, 0.9)
    elif ret_1 > 0.03:
        whale.bias = "bearish"
        whale.last_action = "distribute"
        whale.confidence = min(abs(ret_1) * 8, 0.9)
    agents["whale"] = whale

    # Momentum trader: follows trend
    mom = MarketAgent(archetype="momentum_trader", time_horizon="short")
    if regime.regime == MarketRegime.TREND:
        mom.bias = "bullish" if ret_1 > 0 else "bearish"
        mom.last_action = "chase"
        mom.confidence = 0.7
    else:
        mom.bias = "neutral"
        mom.last_action = "wait"
        mom.confidence = 0.3
    agents["momentum"] = mom

    # Mean reversion trader: fades extremes
    mr = MarketAgent(archetype="mean_reversion_trader", time_horizon="short")
    if rsi_val > 75:
        mr.bias = "bearish"
        mr.last_action = "fade_high"
        mr.confidence = 0.65
    elif rsi_val < 25:
        mr.bias = "bullish"
        mr.last_action = "fade_low"
        mr.confidence = 0.65
    agents["mean_reversion"] = mr

    # Retail FOMO: panics or FOMOs
    retail = MarketAgent(archetype="retail_fomo", time_horizon="short")
    if ret_1 > 0.03 and vol_ratio > 2:
        retail.bias = "bullish"
        retail.last_action = "fomo_buy"
        retail.confidence = 0.8
    elif ret_1 < -0.03:
        retail.bias = "bearish"
        retail.last_action = "panic_sell"
        retail.confidence = 0.7
    agents["retail"] = retail

    # Market maker: provides liquidity, profits from spread
    mm = MarketAgent(archetype="market_maker", time_horizon="short")
    if vol_ratio > 2.5:
        mm.bias = "neutral"
        mm.last_action = "widen_spread"
        mm.confidence = 0.8
    else:
        mm.last_action = "provide_liquidity"
        mm.confidence = 0.6
    agents["market_maker"] = mm

    # Institutional systematic: follows signals
    inst = MarketAgent(archetype="institutional", time_horizon="medium")
    adx_val = float(current.get("adx_14", 0)) if not pd.isna(current.get("adx_14", np.nan)) else 0
    if adx_val > 30:
        inst.bias = "bullish" if ret_1 > 0 else "bearish"
        inst.last_action = "trend_follow"
        inst.confidence = 0.7
    agents["institutional"] = inst

    # Consensus
    bullish = sum(1 for a in agents.values() if a.bias == "bullish")
    bearish = sum(1 for a in agents.values() if a.bias == "bearish")
    neutral = sum(1 for a in agents.values() if a.bias == "neutral")

    return {
        "agents": {k: v.to_dict() for k, v in agents.items()},
        "consensus": {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "net_sentiment": (bullish - bearish) / len(agents) if agents else 0,
        },
    }


# ═══════════════════════════════════════════════════════════════
# HYPOTHESIS GENERATOR FOR AUTORESEARCH
# ═══════════════════════════════════════════════════════════════

def generate_hypotheses(
    regime: RegimeClassification,
    scenarios: list[Scenario],
    agent_consensus: dict,
) -> list[dict]:
    """
    Generate testable hypotheses for autoresearch based on MiroFish analysis.
    Each hypothesis is a structured dict with clear test criteria.
    """
    hypotheses = []

    if regime.regime == MarketRegime.BREAKOUT:
        hypotheses.append({
            "id": "H_BREAKOUT_VOL",
            "statement": "Volatility breakout signals with volume > 2x average produce higher Sharpe than baseline",
            "test": "Compare Sharpe of vol_breakout with volume_surge_threshold=2.0 vs 1.5",
            "priority": "high",
            "strategy": "volatility_breakout",
            "param_changes": {"volume_surge_threshold": [1.5, 2.0, 2.5]},
        })
        hypotheses.append({
            "id": "H_BREAKOUT_CONFIRM",
            "statement": "Requiring 2 confirmation candles reduces false breakouts without excessive lag",
            "test": "Compare win_rate and expectancy with confirmation_candles=1 vs 2 vs 3",
            "priority": "medium",
            "strategy": "volatility_breakout",
            "param_changes": {"breakout_confirmation_candles": [1, 2, 3]},
        })

    if regime.regime == MarketRegime.MEAN_REVERSION:
        hypotheses.append({
            "id": "H_REVERSION_ZSCORE",
            "statement": "Mean reversion entries at z-score > 2.5 have better expectancy than z-score > 2.0",
            "test": "Compare net PnL and profit factor across z-score thresholds",
            "priority": "high",
            "strategy": "mean_reversion",
            "param_changes": {"zscore_entry_threshold": [1.5, 2.0, 2.5, 3.0]},
        })

    if regime.regime == MarketRegime.TREND:
        hypotheses.append({
            "id": "H_TREND_FIB",
            "statement": "Fibonacci 0.618 pullback entries in trending markets outperform 0.382 entries",
            "test": "Compare Sharpe and win rate at different fib levels",
            "priority": "medium",
            "strategy": "trend_following",
            "param_changes": {"fib_levels": [[0.382], [0.5], [0.618], [0.382, 0.5, 0.618]]},
        })

    # Cross-regime hypotheses
    consensus = agent_consensus.get("consensus", {})
    net_sent = consensus.get("net_sentiment", 0)
    if abs(net_sent) > 0.5:
        hypotheses.append({
            "id": "H_CONSENSUS_FILTER",
            "statement": f"Filtering trades by agent consensus (net_sentiment {'>' if net_sent > 0 else '<'} 0.3) improves Sharpe",
            "test": "Add consensus filter to each strategy and compare",
            "priority": "low",
            "strategy": "all",
            "param_changes": {},
        })

    return hypotheses


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_mirofish_analysis(
    df: pd.DataFrame,
    symbol: str = "XRP/USDT",
    bar_idx: int = -1,
    use_llm: bool = True,
    llm_model: str = "llama3.1:8b",
) -> MiroFishReport:
    """
    Run complete MiroFish analysis pipeline.
    Returns a MiroFishReport with regime, scenarios, and hypotheses.

    If use_llm=True and Ollama is running, augments quantitative analysis
    with LLM-generated context, hypotheses, and scenario enrichment.
    """
    # Step 1: Classify regime (always quantitative)
    regime = classify_regime_quantitative(df, bar_idx)

    # Step 2: Generate scenarios (always quantitative)
    scenarios = generate_scenarios_quantitative(df, regime, bar_idx)

    # Step 3: Simulate agent reactions
    agent_sim = simulate_agent_reactions(df, regime, bar_idx)

    # Step 4: Generate hypotheses (deterministic)
    hypotheses = generate_hypotheses(regime, scenarios, agent_sim)

    # Step 4b: LLM augmentation (optional, graceful degradation)
    llm_context = {}
    if use_llm:
        try:
            from mirofish.ollama_engine import (
                is_ollama_running,
                llm_regime_analysis,
                llm_hypothesis_generator,
                llm_scenario_enrichment,
            )

            if is_ollama_running():
                logger.info("Ollama detected -- augmenting with LLM analysis")

                # Gather recent stats for LLM
                current = df.iloc[bar_idx]
                recent_stats = {}
                for col in ["close", "volume_ratio", "rsi_14", "adx_14",
                            "bb_bandwidth", "ret_zscore_20", "realized_vol_5"]:
                    val = current.get(col, None)
                    if val is not None and not pd.isna(val):
                        recent_stats[col] = round(float(val), 4)

                # LLM regime analysis
                llm_regime = llm_regime_analysis(
                    regime.regime.value, regime.confidence,
                    regime.supporting_signals, recent_stats,
                    model=llm_model,
                )
                if llm_regime:
                    llm_context["regime_analysis"] = llm_regime
                    logger.info(f"  LLM regime: {llm_regime.get('regime_assessment', '')[:100]}")

                # LLM extra hypotheses
                scenario_dicts = [s.to_dict() for s in scenarios]
                strategy_names = ["volatility_breakout", "mean_reversion", "trend_following"]
                llm_hyps = llm_hypothesis_generator(
                    regime.regime.value, scenario_dicts,
                    strategy_names, model=llm_model,
                )
                if llm_hyps:
                    hypotheses.extend(llm_hyps)
                    logger.info(f"  LLM added {len(llm_hyps)} hypotheses")

                # LLM scenario enrichment (top scenario only to save time)
                if scenarios:
                    top_scenario = max(scenarios, key=lambda s: s.probability)
                    enrichment = llm_scenario_enrichment(
                        regime.regime.value, top_scenario.name,
                        top_scenario.expected_direction, recent_stats,
                        model=llm_model,
                    )
                    if enrichment:
                        llm_context["scenario_enrichment"] = enrichment
                        logger.info(f"  LLM scenario: {enrichment.get('narrative', '')[:100]}")
            else:
                logger.info("Ollama not running -- using quantitative analysis only")
        except ImportError:
            logger.info("Ollama module not available -- quantitative analysis only")
        except Exception as e:
            logger.warning(f"LLM augmentation failed (non-critical): {e}")

    # Step 5: Determine strategy recommendations
    recommended = []
    avoid = []
    for scenario in scenarios:
        if scenario.probability > 0.3:
            for strat, status in scenario.strategy_implications.items():
                if "ACTIVE" in status and strat not in recommended:
                    recommended.append(strat)
                elif "AVOID" in status and strat not in avoid:
                    avoid.append(strat)

    # Step 6: Risk flags
    risk_flags = []
    if regime.regime == MarketRegime.EXOGENOUS_SHOCK:
        risk_flags.append("EXOGENOUS_SHOCK: Reduce position sizes or halt trading")
    if regime.confidence < 0.3:
        risk_flags.append("LOW_CONFIDENCE: Regime unclear, increase caution")

    consensus = agent_sim.get("consensus", {})
    if abs(consensus.get("net_sentiment", 0)) > 0.7:
        risk_flags.append("EXTREME_CONSENSUS: Crowded positioning risk")

    # Add LLM-identified risks
    if llm_context.get("regime_analysis", {}).get("key_risk"):
        risk_flags.append(f"LLM_RISK: {llm_context['regime_analysis']['key_risk']}")

    report = MiroFishReport(
        timestamp=str(datetime.now(timezone.utc)),
        symbol=symbol,
        regime=regime,
        scenarios=scenarios,
        agent_consensus=agent_sim,
        recommended_strategies=recommended,
        avoid_strategies=avoid,
        risk_flags=risk_flags,
        hypotheses_for_autoresearch=hypotheses,
    )

    return report
