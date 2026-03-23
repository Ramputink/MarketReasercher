"""
CryptoResearchLab -- MiroFish Ollama Integration
LLM-enhanced scenario analysis using local Ollama models.

This module AUGMENTS the quantitative regime classifier with
natural language reasoning. It does NOT replace the deterministic
analysis -- it adds a second opinion layer.

Requires: ollama running locally with llama3.1:8b pulled
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import ollama -- graceful degradation if not available
try:
    import ollama as ollama_sdk
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama SDK not installed. LLM-enhanced analysis disabled.")


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        # Try listing models -- if this works, server is up
        ollama_sdk.list()
        return True
    except Exception:
        return False


def query_ollama(
    prompt: str,
    model: str = "llama3.1:8b",
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    Query local Ollama model.
    Returns response text or None if unavailable.
    """
    if not OLLAMA_AVAILABLE:
        return None

    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama_sdk.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        logger.warning(f"Ollama query failed: {e}")
        return None


# ---------------------------------------------------------------
# LLM-ENHANCED ANALYSIS FUNCTIONS
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative crypto market analyst.
You analyze market data objectively and produce structured assessments.
NEVER give investment advice. Only analyze data patterns.
Always respond in valid JSON format when asked for JSON.
Be concise and data-driven. No speculation beyond what the data shows."""


def llm_regime_analysis(
    regime_name: str,
    confidence: float,
    signals: list,
    recent_stats: dict,
    model: str = "llama3.1:8b",
) -> Optional[dict]:
    """
    Ask LLM to provide deeper regime analysis context.
    Returns structured assessment or None.
    """
    stats_str = "\n".join(f"  - {k}: {v}" for k, v in recent_stats.items())

    prompt = f"""Analyze this crypto market regime classification:

Regime: {regime_name} (confidence: {confidence:.0%})
Supporting signals:
{chr(10).join(f'  - {s}' for s in signals)}

Recent market statistics:
{stats_str}

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "regime_assessment": "1-2 sentence analysis of current regime",
  "key_risk": "biggest risk to monitor right now",
  "expected_duration": "how long this regime typically lasts",
  "transition_probability": "probability of regime change in next 24h (low/medium/high)",
  "volatility_outlook": "increasing/stable/decreasing"
}}"""

    response = query_ollama(prompt, model=model, system=SYSTEM_PROMPT)
    if response is None:
        return None

    try:
        # Try to extract JSON from response
        # Handle cases where LLM wraps in markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Could not parse LLM regime response: {response[:200]}")
        return None


def llm_hypothesis_generator(
    regime_name: str,
    scenarios: list,
    current_strategies: list,
    model: str = "llama3.1:8b",
) -> list:
    """
    Ask LLM to generate additional research hypotheses
    beyond the deterministic ones.
    """
    scenarios_str = "\n".join(
        f"  - {s['name']} (p={s['probability']:.0%}, dir={s['direction']})"
        for s in scenarios
    )

    prompt = f"""Given this crypto market state:

Regime: {regime_name}
Scenarios:
{scenarios_str}

Active strategies: {', '.join(current_strategies)}

Generate 2-3 testable quantitative hypotheses for backtesting.
Each must have measurable criteria.

Respond ONLY with a JSON array (no markdown):
[
  {{
    "id": "H_LLM_1",
    "statement": "hypothesis statement",
    "test": "how to test it with backtesting",
    "priority": "high/medium/low",
    "strategy": "which strategy to modify",
    "param_changes": {{"param_name": [value1, value2]}}
  }}
]"""

    response = query_ollama(prompt, model=model, system=SYSTEM_PROMPT, max_tokens=2048)
    if response is None:
        return []

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        hypotheses = json.loads(cleaned)
        if isinstance(hypotheses, list):
            return hypotheses
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Could not parse LLM hypotheses: {response[:200]}")

    return []


def llm_scenario_enrichment(
    regime_name: str,
    scenario_name: str,
    scenario_direction: str,
    market_stats: dict,
    model: str = "llama3.1:8b",
) -> Optional[dict]:
    """
    Enrich a scenario with deeper LLM-generated context.
    Adds catalysts, risks, and strategy-specific implications.
    """
    stats_str = "\n".join(f"  - {k}: {v}" for k, v in market_stats.items())

    prompt = f"""Analyze this market scenario:

Regime: {regime_name}
Scenario: {scenario_name}
Expected direction: {scenario_direction}
Current stats:
{stats_str}

Provide enriched scenario analysis.

Respond ONLY with a JSON object (no markdown):
{{
  "narrative": "2-3 sentence narrative of what could drive this scenario",
  "catalysts": ["catalyst1", "catalyst2"],
  "invalidation_signals": ["signal that would invalidate this scenario"],
  "timeframe": "expected timeframe for this scenario to play out",
  "confidence_modifier": "increase/decrease/neutral based on current data"
}}"""

    response = query_ollama(prompt, model=model, system=SYSTEM_PROMPT)
    if response is None:
        return None

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Could not parse LLM scenario enrichment: {response[:200]}")
        return None
