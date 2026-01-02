#!/usr/bin/env python3
"""
Kalshi order executor for live trading.

Handles order placement, tracking, and position management on Kalshi.
Compatible with existing TradingEngine interface.
"""
import time
from typing import Dict, Optional
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass

from helpers.kalshi_client import KalshiClient


class OrderSide(Enum):
    """Order side."""
    BUY_YES = "yes"
    BUY_NO = "no"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    RESTING = "resting"  # Open on orderbook
    EXECUTED = "executed"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class KalshiOrder:
    """Order on Kalshi."""
    order_id: Optional[str]
    ticker: str
    side: str  # "yes" or "no"
    quantity: int
    price: int  # Cents
    status: OrderStatus
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


class KalshiExecutor:
    """
    Kalshi order executor.

    Handles order placement and tracking for live trading on Kalshi.
    Mirrors PolymarketExecutor interface for drop-in compatibility.
    """

    def __init__(
        self,
        email: str,
        password: str,
        dry_run: bool = True,
        demo: bool = False
    ):
        """
        Initialize executor.

        Args:
            email: Kalshi account email
            password: Kalshi account password
            dry_run: If True, don't place real orders (default: True)
            demo: Use demo environment
        """
        self.dry_run = dry_run
        self.demo = demo

        # Initialize client
        self.client = KalshiClient(email, password, demo=demo)

        # Order tracking
        self.orders: Dict[str, KalshiOrder] = {}
        self.next_order_id = 1  # For dry_run mode

        # Safety check
        if not dry_run:
            print("⚠️  WARNING: dry_run=False - REAL ORDERS WILL BE PLACED")
            print("   Make sure you understand the risks!")
            confirm = input("Type 'I UNDERSTAND' to continue: ")
            if confirm != "I UNDERSTAND":
                raise Exception("Safety check failed - exiting")

        print(f"✓ Kalshi executor initialized (dry_run={dry_run}, demo={demo})")

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        size_usd: float,
        price: Optional[float] = None,
        order_type: str = "limit"
    ) -> KalshiOrder:
        """
        Place order on Kalshi.

        Args:
            ticker: Market ticker (e.g., "KXBTC-23DEC31-T99999")
            side: OrderSide.BUY_YES or OrderSide.BUY_NO
            size_usd: Order size in USD
            price: Limit price (0.01-0.99), None for market order
            order_type: "limit" or "market"

        Returns:
            KalshiOrder object
        """
        # Convert price to cents
        if price is not None:
            price_cents = int(price * 100)
            price_cents = max(1, min(99, price_cents))  # Clamp to 1-99
        else:
            # Market order - use best price from orderbook
            orderbook = self.client.get_orderbook(ticker, depth=1)
            if side == OrderSide.BUY_YES:
                price_cents = orderbook["yes"][0][0] if orderbook["yes"] else 50
            else:
                price_cents = orderbook["no"][0][0] if orderbook["no"] else 50

        # Calculate quantity (number of contracts)
        # Kalshi: 1 contract = $1 payout if correct
        # Cost per contract = price_cents / 100
        cost_per_contract = price_cents / 100
        quantity = max(1, int(size_usd / cost_per_contract))

        # Create order object
        order = KalshiOrder(
            order_id=None,  # Will be set after placement
            ticker=ticker,
            side=side.value,
            quantity=quantity,
            price=price_cents,
            status=OrderStatus.PENDING,
        )

        print(f"📝 Placing order: {side.value.upper()} {quantity} contracts "
              f"of {ticker} @ {price_cents}¢")

        if self.dry_run:
            # Dry run mode - simulate order
            order.order_id = f"DRY_{self.next_order_id}"
            self.next_order_id += 1
            order.status = OrderStatus.RESTING
            print(f"   [DRY RUN] Order simulated: {order.order_id}")

        else:
            # Real order placement
            try:
                response = self.client.place_order(
                    ticker=ticker,
                    side=side.value,
                    quantity=quantity,
                    price=price_cents,
                    order_type=order_type
                )

                order.order_id = response["order_id"]
                order.status = OrderStatus.RESTING
                print(f"   ✓ Order placed: {order.order_id}")

            except Exception as e:
                print(f"   ❌ Order failed: {e}")
                order.status = OrderStatus.REJECTED
                raise

        # Track order
        self.orders[order.order_id] = order

        return order

    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get current order status.

        Args:
            order_id: Order ID

        Returns:
            OrderStatus
        """
        if order_id not in self.orders:
            raise ValueError(f"Unknown order ID: {order_id}")

        order = self.orders[order_id]

        if self.dry_run:
            # Simulate: assume order fills after 1 second
            elapsed = (datetime.now(timezone.utc) - order.created_at).total_seconds()
            if elapsed > 1 and order.status == OrderStatus.RESTING:
                order.status = OrderStatus.EXECUTED
                order.filled_quantity = order.quantity
                order.avg_fill_price = order.price / 100
                order.updated_at = datetime.now(timezone.utc)

        else:
            # Fetch real order status
            try:
                orders = self.client.get_orders(ticker=order.ticker, limit=100)

                # Find this order
                for api_order in orders:
                    if api_order["order_id"] == order_id:
                        # Update status
                        api_status = api_order["status"]
                        if api_status == "resting":
                            order.status = OrderStatus.RESTING
                        elif api_status == "executed":
                            order.status = OrderStatus.EXECUTED
                            order.filled_quantity = api_order.get("filled_count", order.quantity)
                        elif api_status == "canceled":
                            order.status = OrderStatus.CANCELED
                        order.updated_at = datetime.now(timezone.utc)
                        break

            except Exception as e:
                print(f"⚠️  Failed to get order status: {e}")

        return order.status

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if canceled, False otherwise
        """
        if order_id not in self.orders:
            raise ValueError(f"Unknown order ID: {order_id}")

        order = self.orders[order_id]

        if order.status not in [OrderStatus.PENDING, OrderStatus.RESTING]:
            print(f"⚠️  Cannot cancel order {order_id} - status is {order.status}")
            return False

        print(f"🚫 Canceling order: {order_id}")

        if self.dry_run:
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now(timezone.utc)
            print(f"   [DRY RUN] Order canceled")
            return True

        else:
            try:
                self.client.cancel_order(order_id)
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now(timezone.utc)
                print(f"   ✓ Order canceled")
                return True

            except Exception as e:
                print(f"   ❌ Cancel failed: {e}")
                return False

    def get_fills(self, ticker: Optional[str] = None) -> list:
        """
        Get recent fills.

        Args:
            ticker: Filter by ticker (optional)

        Returns:
            List of fills
        """
        if self.dry_run:
            # Return simulated fills from executed orders
            fills = []
            for order in self.orders.values():
                if order.status == OrderStatus.EXECUTED:
                    if ticker is None or order.ticker == ticker:
                        fills.append({
                            "order_id": order.order_id,
                            "ticker": order.ticker,
                            "side": order.side,
                            "quantity": order.filled_quantity,
                            "price": order.avg_fill_price,
                            "timestamp": order.updated_at.isoformat(),
                        })
            return fills

        else:
            return self.client.get_fills(ticker=ticker)

    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions.

        Returns:
            Dict mapping ticker to position info
        """
        if self.dry_run:
            # Simulate positions from filled orders
            positions = {}
            for order in self.orders.values():
                if order.status == OrderStatus.EXECUTED:
                    if order.ticker not in positions:
                        positions[order.ticker] = {
                            "yes_position": 0,
                            "no_position": 0,
                            "total_cost": 0.0,
                        }

                    if order.side == "yes":
                        positions[order.ticker]["yes_position"] += order.filled_quantity
                    else:
                        positions[order.ticker]["no_position"] += order.filled_quantity

                    positions[order.ticker]["total_cost"] += (
                        order.filled_quantity * order.avg_fill_price
                    )

            return positions

        else:
            api_positions = self.client.get_positions()

            # Convert to dict
            positions = {}
            for pos in api_positions:
                ticker = pos["ticker"]
                positions[ticker] = {
                    "yes_position": pos.get("yes_position", 0),
                    "no_position": pos.get("no_position", 0),
                    "total_cost": pos.get("total_traded", 0) / 100,  # cents → dollars
                }

            return positions

    def get_balance(self) -> float:
        """
        Get account balance in USD.

        Returns:
            Balance in USD
        """
        if self.dry_run:
            return 10000.0  # Simulated $10K balance

        else:
            balance_data = self.client.get_balance()
            return balance_data["balance"] / 100  # cents → dollars


# Example usage
if __name__ == "__main__":
    import os

    email = os.getenv("KALSHI_EMAIL")
    password = os.getenv("KALSHI_PASSWORD")

    if not email or not password:
        print("Set KALSHI_EMAIL and KALSHI_PASSWORD environment variables")
        exit(1)

    # Create executor in dry run mode
    executor = KalshiExecutor(email, password, dry_run=True, demo=False)

    # Get markets
    markets = executor.client.get_markets(status="open", limit=5)

    if markets["markets"]:
        ticker = markets["markets"][0]["ticker"]
        print(f"\nTesting with market: {ticker}")

        # Place test order
        order = executor.place_order(
            ticker=ticker,
            side=OrderSide.BUY_YES,
            size_usd=10.0,
            price=0.55
        )

        print(f"Order ID: {order.order_id}")

        # Wait 2 seconds
        time.sleep(2)

        # Check status
        status = executor.get_order_status(order.order_id)
        print(f"Order status: {status}")

        # Get positions
        positions = executor.get_positions()
        print(f"Positions: {positions}")

        # Get balance
        balance = executor.get_balance()
        print(f"Balance: ${balance:.2f}")
