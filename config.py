"""
CryptoResearchLab -- Configuration
All constants, API settings, strategy defaults, and risk parameters.
Optimized for Mac M2 + Metal GPU + Ollama local LLM.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os


# ---------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------

class MarketRegime(Enum):
    BREAKOUT = "breakout"
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    LATERAL = "lateral"
    EXOGENOUS_SHOCK = "exogenous_shock"
    UNKNOWN = "unknown"


class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class StrategyType(Enum):
    VOLATILITY_BREAKOUT = "volatility_breakout"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"


# ---------------------------------------------------------------
# DATA CONFIG
# ---------------------------------------------------------------

@dataclass
class DataConfig:
    # Primary asset
    symbol: str = "XRP/USDT"
    # Cross-validation assets
    cross_assets: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"
    ])
    # Exchanges
    primary_exchange: str = "binance"
    cross_exchanges: list = field(default_factory=lambda: ["kraken", "coinbase"])
    # Timeframes to collect
    timeframes: list = field(default_factory=lambda: [
        TimeFrame.M1, TimeFrame.M5, TimeFrame.M15,
        TimeFrame.H1, TimeFrame.H4, TimeFrame.D1
    ])
    # Data paths
    data_dir: str = "data"
    # History depth for backtesting (days)
    history_days: int = 365  # Increased from 180: more data = more WF folds = better validation
    # Walk-forward window sizes
    train_window_days: int = 90
    val_window_days: int = 15
    test_window_days: int = 45  # Increased from 30: VB needs ~15 trades per OOS window


# ---------------------------------------------------------------
# BACKTEST CONFIG
# ---------------------------------------------------------------

@dataclass
class BacktestConfig:
    # Commission per trade (one way) -- realistic for crypto
    commission_rate: float = 0.001  # 0.1% taker fee
    # Slippage model
    slippage_bps: float = 5.0  # 5 basis points average slippage
    # Max position size (fraction of equity)
    max_position_pct: float = 0.10
    # Initial capital for simulation
    initial_capital: float = 10_000.0
    # Max holding period (hours)
    max_holding_hours: int = 48
    # Walk-forward folds
    wf_folds: int = 5
    # Minimum trades for statistical significance
    min_trades: int = 10  # Lowered from 30: short walk-forward windows need fewer
    # Minimum Sharpe ratio to accept strategy
    min_sharpe: float = 1.0
    # Maximum drawdown tolerance
    max_drawdown_pct: float = 15.0
    # Out-of-sample degradation tolerance (%)
    max_oos_degradation_pct: float = 50.0  # Relaxed from 30%: low-freq strategies need more room


# ---------------------------------------------------------------
# AUTORESEARCH CONFIG
# ---------------------------------------------------------------

@dataclass
class AutoResearchConfig:
    # Time budget per experiment (seconds)
    experiment_time_budget: int = 300  # 5 minutes
    # Max experiments per session
    max_experiments: int = 50
    # LLM for hypothesis generation (Ollama local)
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434"
    # Experiment log path
    log_dir: str = "logs"
    report_dir: str = "reports"
    # Acceptance criteria: must beat baseline by this margin
    improvement_threshold: float = 0.02  # 2% improvement in primary metric
    # Primary metric for comparison
    primary_metric: str = "sharpe_ratio"
    # Secondary metrics that must not degrade
    guard_metrics: list = field(default_factory=lambda: [
        "max_drawdown", "profit_factor", "oos_sharpe"
    ])


# ---------------------------------------------------------------
# MIROFISH CONFIG
# ---------------------------------------------------------------

@dataclass
class MiroFishConfig:
    # Number of simulated agents in scenario engine
    num_agents: int = 7
    # Agent archetypes for crypto market simulation
    agent_archetypes: list = field(default_factory=lambda: [
        "whale_accumulator",
        "momentum_trader",
        "mean_reversion_trader",
        "news_reactive_trader",
        "market_maker",
        "retail_fomo",
        "institutional_systematic",
    ])
    # Scenario simulation rounds
    simulation_rounds: int = 5
    # News/sentiment sources
    news_keywords: list = field(default_factory=lambda: [
        "XRP", "Ripple", "SEC", "crypto regulation",
        "stablecoin", "CBDC", "DeFi", "whale alert"
    ])
    # Regime classification confidence threshold
    regime_confidence_threshold: float = 0.6
    # LLM settings (Ollama local)
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434"


# ---------------------------------------------------------------
# RISK MANAGEMENT CONFIG
# ---------------------------------------------------------------

@dataclass
class RiskConfig:
    # Max concurrent positions
    max_positions: int = 3
    # Max daily loss (% of equity)
    max_daily_loss_pct: float = 3.0
    # Max drawdown before system halt (% of equity)
    circuit_breaker_drawdown_pct: float = 10.0
    # Position sizing method: "fixed_pct", "kelly", "volatility_scaled"
    sizing_method: str = "volatility_scaled"
    # Kelly fraction (if using kelly)
    kelly_fraction: float = 0.25  # Quarter Kelly
    # Volatility target for vol-scaled sizing
    vol_target_annualized: float = 0.30  # 30% annualized vol target
    # Stop loss (ATR multiplier)
    stop_loss_atr_mult: float = 2.0
    # Take profit (ATR multiplier)
    take_profit_atr_mult: float = 3.0
    # Time stop (hours)
    time_stop_hours: int = 48
    # Min time between trades (minutes) -- anti-overtrading
    min_trade_interval_minutes: int = 15


# ---------------------------------------------------------------
# STRATEGY DEFAULTS
# ---------------------------------------------------------------

@dataclass
class VolatilityBreakoutDefaults:
    bb_period: int = 20
    bb_std: float = 2.0
    bandwidth_percentile_threshold: float = 15.0  # Lowered from 20: detect more compression events
    volume_surge_threshold: float = 2.0  # Raised from 1.5: stronger volume confirmation required
    atr_period: int = 14
    breakout_confirmation_candles: int = 2
    min_range_hours: int = 6  # Raised from 4: require longer consolidation before breakout


@dataclass
class MeanReversionDefaults:
    bb_period: int = 20
    bb_std: float = 2.5
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.5
    rsi_period: int = 14
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    spike_lookback_candles: int = 5  # Raised from 3: more context for spike detection
    min_reversion_pct: float = 0.3  # Lowered from 0.5: easier to trigger on smaller reversions
    # Stop loss uses 3x ATR instead of 2x (wider stops for volatile crypto)
    stop_loss_atr_mult: float = 3.0
    # ADX filter: only trade mean reversion when trend is weak
    max_adx_for_entry: float = 25.0


@dataclass
class TrendFollowingDefaults:
    # Bull Market Support Band (20 SMA + 21 EMA)
    bmsb_sma_period: int = 20
    bmsb_ema_period: int = 21
    # Fibonacci levels for pullback entries
    fib_levels: list = field(default_factory=lambda: [0.382, 0.5, 0.618])
    # Trend confirmation
    adx_period: int = 14
    adx_threshold: float = 25.0
    # EMA for trend direction
    trend_ema_period: int = 50
    # Volume confirmation
    volume_ma_period: int = 20
    min_volume_ratio: float = 1.2


# ---------------------------------------------------------------
# TF MODEL CONFIG (Metal GPU optimized)
# ---------------------------------------------------------------

@dataclass
class TFModelConfig:
    # Architecture
    sequence_length: int = 60       # Bars of history as input
    n_features: int = 0             # Auto-computed from feature pipeline
    hidden_units: list = field(default_factory=lambda: [128, 64, 32])
    lstm_units: int = 64
    dropout_rate: float = 0.3
    # Training
    epochs: int = 100
    batch_size: int = 64            # M2 Metal works well with 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    # Validation
    val_split: float = 0.15
    test_split: float = 0.15
    # Labels -- multi-horizon prediction
    prediction_horizons: list = field(default_factory=lambda: [1, 6, 12, 24])
    label_threshold_pct: float = 0.8  # Min move to classify as up/down (was 0.3, too low for 1h)
    # Metal GPU
    use_mixed_precision: bool = False  # Metal does not support FP16 well yet
    prefetch_buffer: int = 2
    # Model persistence
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"


# ---------------------------------------------------------------
# EXECUTION ENGINE CONFIG
# ---------------------------------------------------------------

@dataclass
class ExecutionConfig:
    # Exchange
    exchange: str = "binance"
    trading_pair: str = "XRP/USDT"
    # Capital
    initial_capital: float = 200.0   # 200 USDC
    # Mode
    paper_trading: bool = True       # Start with paper trading
    # Execution limits
    max_slippage_pct: float = 0.1
    max_order_size_pct: float = 0.15  # Max 15% of equity per trade
    # Cooldowns
    min_trade_interval_seconds: int = 900  # 15 minutes
    # KVM deployment
    kvm_host: str = ""
    kvm_user: str = ""
    kvm_deploy_dir: str = "/opt/cryptolab"


# ---------------------------------------------------------------
# MASTER CONFIG
# ---------------------------------------------------------------

@dataclass
class LabConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    autoresearch: AutoResearchConfig = field(default_factory=AutoResearchConfig)
    mirofish: MiroFishConfig = field(default_factory=MiroFishConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    tf_model: TFModelConfig = field(default_factory=TFModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    vol_breakout: VolatilityBreakoutDefaults = field(default_factory=VolatilityBreakoutDefaults)
    mean_reversion: MeanReversionDefaults = field(default_factory=MeanReversionDefaults)
    trend_following: TrendFollowingDefaults = field(default_factory=TrendFollowingDefaults)
