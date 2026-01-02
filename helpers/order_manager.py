"""
Order state manager for tracking order lifecycle and position reconciliation.

Handles:
- Order state transitions
- Fill tracking and aggregation
- Position reconciliation
- P&L calculation from actual fills
"""
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict

from .polymarket_executor import Order, OrderStatus, OrderSide


@dataclass
class Position:
    """
    Track actual position from filled orders.

    Unlike paper trading, this reflects REAL fills at REAL prices.
    """
    condition_id: str
    asset: str
    side: Optional[str] = None  # "UP" or "DOWN"

    # Position tracking
    total_shares: float = 0.0
    total_cost: float = 0.0  # Total dollars spent
    entry_price: float = 0.0  # Average entry price

    # Order tracking
    active_order_ids: List[str] = field(default_factory=list)
    filled_order_ids: List[str] = field(default_factory=list)

    # Timestamps
    first_entry: Optional[datetime] = None
    last_update: Optional[datetime] = None

    @property
    def has_position(self) -> bool:
        """Check if we have an open position."""
        return self.total_shares > 0

    @property
    def position_value(self) -> float:
        """Current position size in dollars."""
        return self.total_cost

    def add_fill(self, fill_size: float, fill_price: float, timestamp: datetime):
        """
        Add a fill to the position.

        Args:
            fill_size: Size in dollars
            fill_price: Fill price [0, 1]
            timestamp: Fill timestamp
        """
        # Calculate shares from this fill
        shares = fill_size / fill_price if fill_price > 0 else 0

        # Update totals
        self.total_shares += shares
        self.total_cost += fill_size

        # Recalculate average entry price
        if self.total_shares > 0:
            self.entry_price = self.total_cost / self.total_shares

        # Update timestamps
        if not self.first_entry:
            self.first_entry = timestamp
        self.last_update = timestamp

    def close_position(self, exit_price: float) -> float:
        """
        Close position and calculate P&L.

        Args:
            exit_price: Exit price [0, 1]

        Returns:
            Realized P&L in dollars
        """
        if not self.has_position:
            return 0.0

        # Calculate P&L based on shares and price movement
        pnl = (exit_price - self.entry_price) * self.total_shares

        # Reset position
        self.total_shares = 0.0
        self.total_cost = 0.0
        self.entry_price = 0.0
        self.side = None

        return pnl

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.

        Args:
            current_price: Current market price [0, 1]

        Returns:
            Unrealized P&L in dollars
        """
        if not self.has_position:
            return 0.0

        return (current_price - self.entry_price) * self.total_shares


class OrderManager:
    """
    Manage order lifecycle and position reconciliation.

    Key responsibilities:
    - Track order status transitions
    - Aggregate fills into positions
    - Reconcile paper positions with actual fills
    - Calculate P&L from real execution prices
    """

    def __init__(self):
        # Order tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.orders_by_condition: Dict[str, List[str]] = defaultdict(list)

        # Position tracking (by condition_id)
        self.positions: Dict[str, Position] = {}

        # P&L tracking
        self.realized_pnl = 0.0
        self.total_fills = 0
        self.total_orders = 0

        # Stats
        self.fill_latencies: List[float] = []  # Time from submit to fill (seconds)
        self.slippage_values: List[float] = []  # Actual fill vs expected price

    def register_order(self, order: Order, condition_id: str, asset: str):
        """
        Register a new order for tracking.

        Args:
            order: Order object from executor
            condition_id: Market condition ID
            asset: Asset symbol (BTC, ETH, etc.)
        """
        if not order.order_id:
            print("  ⚠ Cannot register order without order_id")
            return

        self.orders[order.order_id] = order
        self.orders_by_condition[condition_id].append(order.order_id)
        self.total_orders += 1

        # Initialize position if needed
        if condition_id not in self.positions:
            self.positions[condition_id] = Position(
                condition_id=condition_id,
                asset=asset
            )

        # Add to active orders
        pos = self.positions[condition_id]
        pos.active_order_ids.append(order.order_id)

        print(f"  📋 Registered order: {order.order_id[:8]}... ({asset})")

    def update_order(self, order: Order, expected_price: Optional[float] = None):
        """
        Update order status and handle fills.

        Args:
            order: Updated order object
            expected_price: Expected fill price for slippage calculation
        """
        order_id = order.order_id
        if order_id not in self.orders:
            print(f"  ⚠ Unknown order: {order_id}")
            return

        old_order = self.orders[order_id]
        old_status = old_order.status
        new_status = order.status

        # Update order
        self.orders[order_id] = order

        # Handle status transitions
        if old_status != new_status:
            self._handle_status_change(order, old_status, new_status)

        # Handle new fills
        if order.filled_size > old_order.filled_size:
            new_fill_size = order.filled_size - old_order.filled_size
            self._handle_fill(order, new_fill_size, expected_price)

    def _handle_status_change(self, order: Order, old_status: OrderStatus, new_status: OrderStatus):
        """Handle order status transitions."""
        if new_status == OrderStatus.FILLED:
            # Calculate fill latency
            if order.submitted_at and order.filled_at:
                latency = (order.filled_at - order.submitted_at).total_seconds()
                self.fill_latencies.append(latency)
                print(f"  ✓ Order filled in {latency:.2f}s: {order.order_id[:8]}...")

        elif new_status == OrderStatus.CANCELLED:
            print(f"  🚫 Order cancelled: {order.order_id[:8]}...")

        elif new_status == OrderStatus.REJECTED:
            print(f"  ✗ Order rejected: {order.order_id[:8]}...")

    def _handle_fill(self, order: Order, new_fill_size: float, expected_price: Optional[float]):
        """
        Process a new fill.

        Args:
            order: Order that was filled
            new_fill_size: Size of new fill in dollars
            expected_price: Expected price for slippage calc
        """
        # Find position
        condition_id = None
        for cid, order_ids in self.orders_by_condition.items():
            if order.order_id in order_ids:
                condition_id = cid
                break

        if not condition_id:
            print(f"  ⚠ Cannot find condition for order: {order.order_id}")
            return

        pos = self.positions[condition_id]

        # Calculate fill price (approximate from average)
        fill_price = order.average_fill_price if order.average_fill_price > 0 else order.price

        # Track slippage if we have expected price
        if expected_price:
            slippage = fill_price - expected_price
            self.slippage_values.append(slippage)
            slippage_pct = (slippage / expected_price) * 100
            print(f"  💸 Slippage: {slippage:+.4f} ({slippage_pct:+.2f}%)")

        # Update position
        fill_timestamp = order.filled_at or datetime.now(timezone.utc)
        pos.add_fill(new_fill_size, fill_price, fill_timestamp)
        pos.side = "UP" if order.side == OrderSide.BUY else "DOWN"

        self.total_fills += 1

        print(f"  📊 Fill: +{new_fill_size:.2f} @ {fill_price:.3f} | "
              f"Total: {pos.total_shares:.2f} shares @ {pos.entry_price:.3f} avg")

    def close_position(self, condition_id: str, exit_price: float) -> Tuple[float, Position]:
        """
        Close position and calculate realized P&L.

        Args:
            condition_id: Market to close
            exit_price: Exit price [0, 1]

        Returns:
            (realized_pnl, position)
        """
        if condition_id not in self.positions:
            return 0.0, None

        pos = self.positions[condition_id]
        pnl = pos.close_position(exit_price)
        self.realized_pnl += pnl

        # Move orders to filled list
        for order_id in pos.active_order_ids:
            if order_id not in pos.filled_order_ids:
                pos.filled_order_ids.append(order_id)
        pos.active_order_ids.clear()

        print(f"  💰 Position closed: {pnl:+.2f} PnL | Total PnL: ${self.realized_pnl:+.2f}")

        return pnl, pos

    def reconcile_position(self, condition_id: str, paper_position) -> Dict:
        """
        Reconcile paper trading position with actual fills.

        Compares expected (paper) vs actual (fills) to detect discrepancies.

        Args:
            condition_id: Market to reconcile
            paper_position: Paper trading Position object

        Returns:
            Dict with reconciliation results
        """
        if condition_id not in self.positions:
            return {
                "status": "no_position",
                "discrepancy": False
            }

        actual_pos = self.positions[condition_id]

        # Compare sizes
        paper_size = paper_position.size if paper_position else 0
        actual_size = actual_pos.position_value
        size_diff = abs(actual_size - paper_size)

        # Compare prices
        paper_price = paper_position.entry_price if paper_position else 0
        actual_price = actual_pos.entry_price
        price_diff = abs(actual_price - paper_price)

        # Detect discrepancies
        discrepancy = size_diff > 1.0 or price_diff > 0.05  # Thresholds

        result = {
            "status": "ok" if not discrepancy else "discrepancy",
            "discrepancy": discrepancy,
            "paper_size": paper_size,
            "actual_size": actual_size,
            "size_diff": size_diff,
            "paper_price": paper_price,
            "actual_price": actual_price,
            "price_diff": price_diff,
        }

        if discrepancy:
            print(f"  ⚠ POSITION DISCREPANCY for {condition_id}:")
            print(f"    Paper: ${paper_size:.2f} @ {paper_price:.3f}")
            print(f"    Actual: ${actual_size:.2f} @ {actual_price:.3f}")
            print(f"    Diff: ${size_diff:.2f} size, {price_diff:.3f} price")

        return result

    def get_stats(self) -> Dict:
        """Get order execution statistics."""
        avg_fill_latency = sum(self.fill_latencies) / len(self.fill_latencies) if self.fill_latencies else 0
        avg_slippage = sum(self.slippage_values) / len(self.slippage_values) if self.slippage_values else 0

        fill_rate = self.total_fills / max(1, self.total_orders) * 100

        return {
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "fill_rate": fill_rate,
            "avg_fill_latency_sec": avg_fill_latency,
            "avg_slippage": avg_slippage,
            "realized_pnl": self.realized_pnl,
            "active_positions": sum(1 for p in self.positions.values() if p.has_position)
        }

    def print_stats(self):
        """Print execution statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("ORDER EXECUTION STATS")
        print("=" * 60)
        print(f"Total orders: {stats['total_orders']}")
        print(f"Total fills: {stats['total_fills']} ({stats['fill_rate']:.1f}% fill rate)")
        print(f"Avg fill latency: {stats['avg_fill_latency_sec']:.2f}s")
        print(f"Avg slippage: {stats['avg_slippage']:+.4f}")
        print(f"Realized P&L: ${stats['realized_pnl']:+.2f}")
        print(f"Active positions: {stats['active_positions']}")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    from polymarket_executor import Order, OrderSide, OrderStatus

    print("=" * 60)
    print("Order Manager - Example Usage")
    print("=" * 60)

    manager = OrderManager()

    # Simulate an order lifecycle
    order = Order(
        token_id="0x1234",
        side=OrderSide.BUY,
        price=0.45,
        size=10.0,
        order_id="test_order_001"
    )

    # Register
    manager.register_order(order, "condition_123", "BTC")

    # Update to filled
    order.status = OrderStatus.FILLED
    order.filled_size = 10.0
    order.average_fill_price = 0.46  # Slight slippage
    order.filled_at = datetime.now(timezone.utc)

    manager.update_order(order, expected_price=0.45)

    # Close position
    pnl, pos = manager.close_position("condition_123", exit_price=0.50)

    # Print stats
    manager.print_stats()
