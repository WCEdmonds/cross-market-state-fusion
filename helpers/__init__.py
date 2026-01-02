"""
Polymarket and Kalshi trading helpers.
"""
from .polymarket_api import (
    get_15m_markets,
    get_next_market,
    Market,
)
from .binance_wss import (
    BinanceStreamer,
    get_current_prices,
)
from .orderbook_wss import (
    OrderbookStreamer,
)
from .binance_futures import (
    FuturesStreamer,
    FuturesState,
    get_futures_snapshot,
)
from .training_logger import (
    TrainingLogger,
    get_logger,
    reset_logger,
)

# Kalshi integration (for U.S. users)
try:
    from .kalshi_client import KalshiClient, KalshiMarket
    from .kalshi_executor import KalshiExecutor, OrderSide as KalshiOrderSide
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    KalshiClient = None
    KalshiMarket = None
    KalshiExecutor = None
    KalshiOrderSide = None

# Backwards compat
get_active_markets = get_15m_markets

__all__ = [
    # Polymarket
    "get_15m_markets",
    "get_active_markets",
    "get_next_market",
    "Market",
    # Market data
    "BinanceStreamer",
    "get_current_prices",
    "OrderbookStreamer",
    "FuturesStreamer",
    "FuturesState",
    "get_futures_snapshot",
    # Training
    "TrainingLogger",
    "get_logger",
    "reset_logger",
    # Kalshi (U.S. legal)
    "KalshiClient",
    "KalshiMarket",
    "KalshiExecutor",
    "KalshiOrderSide",
    "KALSHI_AVAILABLE",
]
