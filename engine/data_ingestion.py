"""
CryptoResearchLab -- Data Ingestion Engine
Fetches OHLCV, orderbook, funding, and market data via ccxt.
Stores locally as parquet for fast backtesting.
Credentials loaded from .env via env_loader.
"""
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd
import numpy as np

from config import DataConfig, TimeFrame
from engine.env_loader import get_binance_credentials, has_binance_keys

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# TIMEFRAME HELPERS
# ═══════════════════════════════════════════════════════════════

TF_TO_MS = {
    TimeFrame.M1: 60_000,
    TimeFrame.M5: 300_000,
    TimeFrame.M15: 900_000,
    TimeFrame.H1: 3_600_000,
    TimeFrame.H4: 14_400_000,
    TimeFrame.D1: 86_400_000,
}

TF_TO_CCXT = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "4h",
    TimeFrame.D1: "1d",
}


class DataIngestionEngine:
    """
    Handles all market data collection and storage.
    Supports multiple exchanges and symbols for cross-validation.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.exchanges: dict[str, ccxt.Exchange] = {}
        self._init_exchanges()

    def _init_exchanges(self):
        """Initialize exchange connections with credentials from .env."""
        exchange_classes = {
            "binance": ccxt.binance,
            "kraken": ccxt.kraken,
            "coinbase": ccxt.coinbasepro if hasattr(ccxt, 'coinbasepro') else ccxt.coinbase,
        }

        # Load Binance credentials from .env
        binance_creds = get_binance_credentials()

        for name in [self.config.primary_exchange] + self.config.cross_exchanges:
            if name in exchange_classes:
                try:
                    exchange_config = {
                        "enableRateLimit": True,
                        "timeout": 30000,
                    }
                    # Inject credentials for Binance
                    if name == "binance" and binance_creds.get("apiKey"):
                        exchange_config["apiKey"] = binance_creds["apiKey"]
                        exchange_config["secret"] = binance_creds["secret"]
                        # Disable fetchCurrencies — it calls /sapi/v1/capital/config/getall
                        # which requires extra permissions we don't need for OHLCV (public data)
                        exchange_config["options"] = {
                            "fetchCurrencies": False,
                            "warnOnFetchCurrenciesWithoutSupport": False,
                        }
                        logger.info(f"Connecting to {name} with API key (read-only)")
                    else:
                        logger.info(f"Connecting to {name} (public endpoints)")

                    self.exchanges[name] = exchange_classes[name](exchange_config)
                    logger.info(f"Connected to {name}")
                except Exception as e:
                    logger.warning(f"Failed to connect to {name}: {e}")

    def _parquet_path(self, exchange: str, symbol: str, tf: TimeFrame) -> str:
        """Generate parquet file path for cached data."""
        safe_symbol = symbol.replace("/", "_")
        safe_tf = tf.value
        os.makedirs(self.config.data_dir, exist_ok=True)
        return os.path.join(
            self.config.data_dir,
            f"{exchange}_{safe_symbol}_{safe_tf}.parquet"
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        exchange_name: Optional[str] = None,
        since_days: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Uses local cache if available and fresh.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        exchange_name = exchange_name or self.config.primary_exchange
        since_days = since_days or self.config.history_days
        path = self._parquet_path(exchange_name, symbol, timeframe)

        # Check cache
        if not force_refresh and os.path.exists(path):
            cached = pd.read_parquet(path)
            if len(cached) > 0:
                last_ts = cached["timestamp"].max()
                age_hours = (
                    datetime.now(timezone.utc) - pd.Timestamp(last_ts, unit="ms", tz="UTC")
                ).total_seconds() / 3600
                # Cache is fresh if less than 1 candle old
                tf_hours = TF_TO_MS[timeframe] / 3_600_000
                if age_hours < tf_hours * 2:
                    logger.info(
                        f"Using cached {symbol} {timeframe.value} from {exchange_name} "
                        f"({len(cached)} candles)"
                    )
                    return cached

        # Fetch from exchange
        exchange = self.exchanges.get(exchange_name)
        if exchange is None:
            raise ValueError(f"Exchange {exchange_name} not available")

        since_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000
        )
        ccxt_tf = TF_TO_CCXT[timeframe]
        all_candles = []
        current_since = since_ms
        limit = 1000

        logger.info(
            f"Fetching {symbol} {timeframe.value} from {exchange_name} "
            f"(last {since_days} days)..."
        )

        while True:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, ccxt_tf, since=current_since, limit=limit
                )
            except Exception as e:
                logger.error(f"Error fetching {symbol} from {exchange_name}: {e}")
                break

            if not candles:
                break

            all_candles.extend(candles)
            current_since = candles[-1][0] + TF_TO_MS[timeframe]

            if len(candles) < limit:
                break

            time.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            # Fallback: retry without API keys (public endpoint only)
            if exchange_name == "binance" and exchange.apiKey:
                logger.warning(
                    f"Authenticated fetch failed for {symbol}. "
                    f"Retrying with public endpoint (no API keys)..."
                )
                try:
                    pub_exchange = ccxt.binance({
                        "enableRateLimit": True,
                        "timeout": 30000,
                    })
                    pub_since = since_ms
                    while True:
                        try:
                            candles = pub_exchange.fetch_ohlcv(
                                symbol, ccxt_tf, since=pub_since, limit=limit
                            )
                        except Exception as e2:
                            logger.error(f"Public fallback also failed: {e2}")
                            break
                        if not candles:
                            break
                        all_candles.extend(candles)
                        pub_since = candles[-1][0] + TF_TO_MS[timeframe]
                        if len(candles) < limit:
                            break
                        time.sleep(pub_exchange.rateLimit / 1000)
                except Exception as e3:
                    logger.error(f"Public fallback init failed: {e3}")

        if not all_candles:
            logger.warning(f"No data fetched for {symbol} from {exchange_name}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Save cache
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} candles to {path}")

        return df

    def fetch_funding_rate(
        self, symbol: str, exchange_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch funding rate history (perpetual futures)."""
        exchange_name = exchange_name or self.config.primary_exchange
        exchange = self.exchanges.get(exchange_name)
        if exchange is None or not hasattr(exchange, "fetch_funding_rate_history"):
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        try:
            since_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000
            )
            data = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
            if data:
                df = pd.DataFrame(data)
                df = df[["timestamp", "fundingRate"]].rename(
                    columns={"fundingRate": "funding_rate"}
                )
                return df
        except Exception as e:
            logger.warning(f"Could not fetch funding rates: {e}")

        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    def fetch_orderbook_snapshot(
        self, symbol: str, exchange_name: Optional[str] = None, depth: int = 20
    ) -> dict:
        """Fetch current orderbook snapshot."""
        exchange_name = exchange_name or self.config.primary_exchange
        exchange = self.exchanges.get(exchange_name)
        if exchange is None:
            return {"bids": [], "asks": [], "timestamp": None}

        try:
            ob = exchange.fetch_order_book(symbol, limit=depth)
            return {
                "bids": ob.get("bids", []),
                "asks": ob.get("asks", []),
                "timestamp": ob.get("timestamp"),
                "bid_volume": sum(b[1] for b in ob.get("bids", [])),
                "ask_volume": sum(a[1] for a in ob.get("asks", [])),
                "spread_bps": (
                    (ob["asks"][0][0] - ob["bids"][0][0]) / ob["bids"][0][0] * 10000
                    if ob.get("bids") and ob.get("asks") else None
                ),
            }
        except Exception as e:
            logger.warning(f"Could not fetch orderbook: {e}")
            return {"bids": [], "asks": [], "timestamp": None}

    def fetch_all_primary(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """
        Fetch all timeframes for the primary symbol on the primary exchange.
        Returns dict keyed by timeframe value string.
        """
        result = {}
        for tf in self.config.timeframes:
            df = self.fetch_ohlcv(
                self.config.symbol, tf,
                exchange_name=self.config.primary_exchange,
                force_refresh=force_refresh,
            )
            result[tf.value] = df
        return result

    def fetch_cross_validation_data(
        self, timeframe: TimeFrame = TimeFrame.H1
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for cross-asset and cross-exchange validation.
        Returns dict keyed by "exchange_symbol".
        """
        result = {}
        # Cross-asset on primary exchange
        for symbol in self.config.cross_assets:
            key = f"{self.config.primary_exchange}_{symbol.replace('/', '_')}"
            try:
                df = self.fetch_ohlcv(symbol, timeframe, self.config.primary_exchange)
                if len(df) > 0:
                    result[key] = df
            except Exception as e:
                logger.warning(f"Could not fetch {symbol}: {e}")

        # Same asset on different exchanges
        for ex_name in self.config.cross_exchanges:
            key = f"{ex_name}_{self.config.symbol.replace('/', '_')}"
            try:
                df = self.fetch_ohlcv(self.config.symbol, timeframe, ex_name)
                if len(df) > 0:
                    result[key] = df
            except Exception as e:
                logger.warning(f"Could not fetch from {ex_name}: {e}")

        return result



# NOTE: generate_synthetic_data was removed intentionally.
# All training and backtesting now uses real Binance market data exclusively.
# API keys must be configured in .env before running the pipeline.
