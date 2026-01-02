#!/usr/bin/env python3
"""
Kalshi API client.

Official API docs: https://trading-api.readme.io/reference/getting-started

Kalshi is a CFTC-regulated prediction market exchange, legal for U.S. users.
"""
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class KalshiMarket:
    """Kalshi market information."""
    ticker: str
    title: str
    subtitle: str
    close_time: datetime
    expiration_time: datetime
    strike_type: str  # "binary", "scalar", etc.
    floor_strike: float
    cap_strike: float
    status: str  # "open", "closed", "settled"
    yes_bid: float  # Best YES bid (cents)
    yes_ask: float  # Best YES ask (cents)
    no_bid: float   # Best NO bid (cents)
    no_ask: float   # Best NO ask (cents)
    volume: float
    open_interest: int


class KalshiClient:
    """
    Kalshi REST API client.

    Handles authentication, market data, and order management.
    """

    def __init__(self, email: str, password: str, demo: bool = False):
        """
        Initialize Kalshi client.

        Args:
            email: Kalshi account email
            password: Kalshi account password
            demo: Use demo environment (default: False)
        """
        self.email = email
        self.password = password

        # API endpoints
        if demo:
            self.base_url = "https://demo-api.kalshi.co/trade-api/v2"
        else:
            self.base_url = "https://trading-api.kalshi.com/trade-api/v2"

        # Auth token
        self.token = None
        self.token_expiry = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Login
        self.login()

    def _wait_for_rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _ensure_authenticated(self):
        """Refresh token if expired."""
        if self.token_expiry and datetime.now(timezone.utc) >= self.token_expiry:
            self.login()

    def login(self):
        """
        Authenticate with Kalshi API.

        Returns auth token valid for 24 hours.
        """
        self._wait_for_rate_limit()

        response = requests.post(
            f"{self.base_url}/login",
            json={"email": self.email, "password": self.password}
        )

        if response.status_code != 200:
            raise Exception(f"Login failed: {response.text}")

        data = response.json()
        self.token = data["token"]

        # Token expires in 24 hours
        self.token_expiry = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).replace(day=datetime.now(timezone.utc).day + 1)

        print(f"✓ Kalshi authenticated (token expires: {self.token_expiry})")

    def _get_headers(self) -> Dict[str, str]:
        """Get auth headers for API requests."""
        self._ensure_authenticated()
        return {"Authorization": f"Bearer {self.token}"}

    def get_exchange_status(self) -> Dict:
        """Get exchange status."""
        self._wait_for_rate_limit()

        response = requests.get(
            f"{self.base_url}/exchange/status",
            headers=self._get_headers()
        )

        return response.json()

    def get_markets(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None
    ) -> Dict:
        """
        Get markets.

        Args:
            status: Market status ("open", "closed", "settled", or "all")
            series_ticker: Filter by series (e.g., "KXBTC")
            limit: Max markets to return (max 200)
            cursor: Pagination cursor

        Returns:
            {
                "cursor": "next_cursor",
                "markets": [...]
            }
        """
        self._wait_for_rate_limit()

        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            f"{self.base_url}/markets",
            headers=self._get_headers(),
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Get markets failed: {response.text}")

        return response.json()

    def get_market(self, ticker: str) -> Dict:
        """
        Get single market by ticker.

        Args:
            ticker: Market ticker (e.g., "KXBTC-23DEC31-T99999")

        Returns:
            Market details
        """
        self._wait_for_rate_limit()

        response = requests.get(
            f"{self.base_url}/markets/{ticker}",
            headers=self._get_headers()
        )

        if response.status_code != 200:
            raise Exception(f"Get market failed: {response.text}")

        return response.json()["market"]

    def get_orderbook(self, ticker: str, depth: int = 5) -> Dict:
        """
        Get orderbook for market.

        Args:
            ticker: Market ticker
            depth: Order book depth (default: 5 levels)

        Returns:
            {
                "yes": [[price, quantity], ...],  # Bids
                "no": [[price, quantity], ...]     # Asks
            }
        """
        self._wait_for_rate_limit()

        response = requests.get(
            f"{self.base_url}/markets/{ticker}/orderbook",
            headers=self._get_headers(),
            params={"depth": depth}
        )

        if response.status_code != 200:
            raise Exception(f"Get orderbook failed: {response.text}")

        data = response.json()["orderbook"]

        # Kalshi format: {yes: [[price, qty], ...], no: [[price, qty], ...]}
        return {
            "yes": data.get("yes", []),
            "no": data.get("no", [])
        }

    def get_trades(self, ticker: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades for market.

        Args:
            ticker: Market ticker
            limit: Max trades to return

        Returns:
            List of recent trades
        """
        self._wait_for_rate_limit()

        response = requests.get(
            f"{self.base_url}/markets/{ticker}/trades",
            headers=self._get_headers(),
            params={"limit": limit}
        )

        if response.status_code != 200:
            raise Exception(f"Get trades failed: {response.text}")

        return response.json()["trades"]

    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: int,
        order_type: str = "limit",
        expiration_ts: Optional[int] = None
    ) -> Dict:
        """
        Place order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            quantity: Number of contracts
            price: Price in cents (1-99)
            order_type: "limit" or "market"
            expiration_ts: Order expiration timestamp (optional)

        Returns:
            Order details
        """
        self._wait_for_rate_limit()
        self._ensure_authenticated()

        # Validate inputs
        if side not in ["yes", "no"]:
            raise ValueError(f"Invalid side: {side}. Must be 'yes' or 'no'")

        if not (1 <= price <= 99):
            raise ValueError(f"Invalid price: {price}. Must be 1-99 cents")

        # Build order payload
        payload = {
            "ticker": ticker,
            "type": order_type,
            "side": side,
            "count": quantity,
        }

        # Add price for limit orders
        if order_type == "limit":
            if side == "yes":
                payload["yes_price"] = price
            else:
                payload["no_price"] = price

        if expiration_ts:
            payload["expiration_ts"] = expiration_ts

        response = requests.post(
            f"{self.base_url}/portfolio/orders",
            headers=self._get_headers(),
            json=payload
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Place order failed: {response.text}")

        return response.json()["order"]

    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation confirmation
        """
        self._wait_for_rate_limit()

        response = requests.delete(
            f"{self.base_url}/portfolio/orders/{order_id}",
            headers=self._get_headers()
        )

        if response.status_code != 200:
            raise Exception(f"Cancel order failed: {response.text}")

        return response.json()

    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get user's orders.

        Args:
            ticker: Filter by ticker (optional)
            status: Filter by status ("resting", "canceled", "executed")
            limit: Max orders to return

        Returns:
            List of orders
        """
        self._wait_for_rate_limit()

        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        response = requests.get(
            f"{self.base_url}/portfolio/orders",
            headers=self._get_headers(),
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Get orders failed: {response.text}")

        return response.json()["orders"]

    def get_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get user's fills.

        Args:
            ticker: Filter by ticker (optional)
            limit: Max fills to return

        Returns:
            List of fills
        """
        self._wait_for_rate_limit()

        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker

        response = requests.get(
            f"{self.base_url}/portfolio/fills",
            headers=self._get_headers(),
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Get fills failed: {response.text}")

        return response.json()["fills"]

    def get_positions(self) -> List[Dict]:
        """
        Get user's current positions.

        Returns:
            List of positions
        """
        self._wait_for_rate_limit()

        response = requests.get(
            f"{self.base_url}/portfolio/positions",
            headers=self._get_headers()
        )

        if response.status_code != 200:
            raise Exception(f"Get positions failed: {response.text}")

        return response.json()["positions"]

    def get_balance(self) -> Dict:
        """
        Get account balance.

        Returns:
            {
                "balance": <cents>,
                "payout": <cents>
            }
        """
        self._wait_for_rate_limit()

        response = requests.get(
            f"{self.base_url}/portfolio/balance",
            headers=self._get_headers()
        )

        if response.status_code != 200:
            raise Exception(f"Get balance failed: {response.text}")

        return response.json()


# Example usage
if __name__ == "__main__":
    import os

    # Get credentials from environment
    email = os.getenv("KALSHI_EMAIL")
    password = os.getenv("KALSHI_PASSWORD")

    if not email or not password:
        print("Set KALSHI_EMAIL and KALSHI_PASSWORD environment variables")
        exit(1)

    # Create client
    client = KalshiClient(email, password, demo=False)

    # Check exchange status
    status = client.get_exchange_status()
    print(f"Exchange status: {status}")

    # Get open markets
    markets_data = client.get_markets(status="open", limit=10)
    markets = markets_data.get("markets", [])
    print(f"\nFound {len(markets)} open markets")

    for market in markets[:5]:
        print(f"  • {market['ticker']}: {market['title']}")

    # Get orderbook for first market
    if markets:
        ticker = markets[0]["ticker"]
        orderbook = client.get_orderbook(ticker)
        print(f"\nOrderbook for {ticker}:")
        print(f"  YES bids: {orderbook['yes'][:3]}")
        print(f"  NO bids: {orderbook['no'][:3]}")

    # Get balance
    balance = client.get_balance()
    print(f"\nAccount balance: ${balance['balance'] / 100:.2f}")
