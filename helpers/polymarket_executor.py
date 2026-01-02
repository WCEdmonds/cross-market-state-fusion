"""
Polymarket CLOB order executor for live trading.

Handles:
- Order creation with EIP-712 signatures
- Order placement via REST API
- Order status tracking and updates
- Position reconciliation

IMPORTANT: This enables REAL MONEY trading. Use with extreme caution.
"""
import time
import requests
import hashlib
import json
from typing import Optional, Dict, List, Literal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Try to import web3 for signing (optional)
try:
    from eth_account import Account
    from eth_account.messages import encode_structured_data
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Warning: eth_account not installed. Install with: pip install eth-account")

CLOB_API = "https://clob.polymarket.com"


class OrderSide(Enum):
    """Order side: BUY or SELL."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"       # Created locally, not yet sent
    SUBMITTED = "submitted"   # Sent to CLOB, waiting for ack
    OPEN = "open"            # Active on orderbook
    MATCHED = "matched"      # Matched but not settled
    FILLED = "filled"        # Fully filled
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Polymarket CLOB order."""
    # User-specified
    token_id: str              # Token to trade (UP or DOWN token ID)
    side: OrderSide           # BUY or SELL
    price: float              # Limit price [0, 1]
    size: float               # Size in dollars

    # Polymarket-specific
    salt: int = field(default_factory=lambda: int(time.time() * 1000000))
    expiration: int = field(default_factory=lambda: int(time.time()) + 86400)  # 24h default
    nonce: int = 0
    fee_rate_bps: int = 0     # Fee in basis points

    # Metadata
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Fill tracking
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)

    # Signature
    signature: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is active (can still fill)."""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.MATCHED]

    @property
    def remaining_size(self) -> float:
        """Unfilled size."""
        return max(0, self.size - self.filled_size)


class PolymarketExecutor:
    """
    Execute orders on Polymarket CLOB.

    Handles order signing, submission, tracking, and reconciliation.

    SECURITY WARNING:
    - Never commit private keys to version control
    - Use environment variables or secure key management
    - Test with small sizes before scaling up
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        chain_id: int = 137,  # Polygon mainnet
        api_creds: Optional[Dict[str, str]] = None,
        dry_run: bool = True
    ):
        """
        Initialize executor.

        Args:
            private_key: Ethereum private key (hex string, with or without 0x)
            chain_id: Polygon chain ID (137 for mainnet, 80001 for Mumbai testnet)
            api_creds: Optional API key/secret for Polymarket (if required)
            dry_run: If True, log orders but don't submit (safety default)
        """
        self.dry_run = dry_run
        self.chain_id = chain_id
        self.api_creds = api_creds or {}

        # Private key handling
        self.private_key = None
        self.address = None

        if private_key and WEB3_AVAILABLE:
            # Remove 0x prefix if present
            pk = private_key[2:] if private_key.startswith('0x') else private_key
            self.private_key = pk
            account = Account.from_key(pk)
            self.address = account.address
            print(f"✓ Executor initialized for address: {self.address}")
        elif not WEB3_AVAILABLE:
            print("✗ eth_account not available - install with: pip install eth-account")
        else:
            print("⚠ No private key provided - executor in read-only mode")

        if dry_run:
            print("⚠ DRY RUN MODE - Orders will be logged but NOT submitted")

        # Order tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.active_orders: List[str] = []  # List of active order IDs

        # HTTP session for connection pooling (latency optimization)
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'PolymarketExecutor/1.0'
        })

    def _generate_order_hash(self, order: Order, maker_address: str) -> str:
        """
        Generate EIP-712 typed data hash for Polymarket order.

        Polymarket uses a specific EIP-712 structure for order signing.
        This is a simplified version - actual implementation needs the
        exact domain separator and type hash from Polymarket docs.

        TODO: Update with actual Polymarket EIP-712 schema
        """
        if not WEB3_AVAILABLE:
            raise RuntimeError("eth_account required for order signing")

        # Polymarket EIP-712 domain (UPDATE THESE VALUES)
        domain = {
            "name": "Polymarket CTF Exchange",
            "version": "1",
            "chainId": self.chain_id,
            "verifyingContract": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # EXAMPLE - UPDATE
        }

        # Order type (UPDATE THIS STRUCTURE)
        types = {
            "Order": [
                {"name": "maker", "type": "address"},
                {"name": "taker", "type": "address"},
                {"name": "tokenId", "type": "uint256"},
                {"name": "makerAmount", "type": "uint256"},
                {"name": "takerAmount", "type": "uint256"},
                {"name": "side", "type": "uint8"},
                {"name": "expiration", "type": "uint256"},
                {"name": "nonce", "type": "uint256"},
                {"name": "feeRateBps", "type": "uint256"},
                {"name": "signatureType", "type": "uint8"},
                {"name": "salt", "type": "uint256"}
            ]
        }

        # Convert size and price to amounts (UPDATE CALCULATION)
        # This is approximate - actual conversion depends on Polymarket's contract
        maker_amount = int(order.size * 1e6)  # Convert to USDC (6 decimals)
        taker_amount = int((order.size / order.price) * 1e6)

        message = {
            "maker": maker_address,
            "taker": "0x0000000000000000000000000000000000000000",  # Any taker
            "tokenId": int(order.token_id, 16),  # Token ID as uint256
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "side": 0 if order.side == OrderSide.BUY else 1,
            "expiration": order.expiration,
            "nonce": order.nonce,
            "feeRateBps": order.fee_rate_bps,
            "signatureType": 0,  # EOA signature
            "salt": order.salt
        }

        structured_data = {
            "types": types,
            "primaryType": "Order",
            "domain": domain,
            "message": message
        }

        return encode_structured_data(structured_data)

    def sign_order(self, order: Order) -> Order:
        """
        Sign order with EIP-712 signature.

        Args:
            order: Order to sign

        Returns:
            Order with signature field populated
        """
        if not self.private_key or not self.address:
            raise RuntimeError("Private key required for signing orders")

        # Generate typed data hash
        signable_message = self._generate_order_hash(order, self.address)

        # Sign
        account = Account.from_key(self.private_key)
        signed = account.sign_message(signable_message)

        # Signature as hex string
        order.signature = signed.signature.hex()

        print(f"  ✓ Order signed: {order.side.value} {order.size} @ {order.price:.3f}")
        return order

    def create_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        post_only: bool = True,
        expiration: Optional[int] = None
    ) -> Order:
        """
        Create and sign a new order.

        Args:
            token_id: Token ID to trade (UP or DOWN token)
            side: BUY or SELL
            price: Limit price [0, 1]
            size: Size in dollars
            post_only: If True, order will only add liquidity (maker order)
            expiration: Unix timestamp when order expires (default: 24h)

        Returns:
            Signed Order ready for submission
        """
        # Validate inputs
        if not 0 < price < 1:
            raise ValueError(f"Price must be in (0, 1), got {price}")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        # Create order
        order = Order(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            expiration=expiration or (int(time.time()) + 86400),
        )

        # Sign if we have a key
        if self.private_key:
            order = self.sign_order(order)

        return order

    def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit order to Polymarket CLOB.

        Args:
            order: Signed order to submit

        Returns:
            Order ID if successful, None otherwise
        """
        if self.dry_run:
            # Dry run: log but don't submit
            order_id = f"DRY_{order.salt}"
            order.order_id = order_id
            order.status = OrderStatus.OPEN
            order.submitted_at = datetime.now(timezone.utc)
            self.orders[order_id] = order
            self.active_orders.append(order_id)

            print(f"  [DRY RUN] Would submit: {order.side.value} {order.size} @ {order.price:.3f}")
            return order_id

        if not order.signature:
            raise RuntimeError("Order must be signed before submission")

        # Build request payload (UPDATE WITH ACTUAL POLYMARKET API SCHEMA)
        payload = {
            "tokenID": order.token_id,
            "price": str(order.price),
            "size": str(order.size),
            "side": order.side.value,
            "signature": order.signature,
            "salt": order.salt,
            "expiration": order.expiration,
            "nonce": order.nonce,
            "feeRateBps": order.fee_rate_bps,
            "maker": self.address,
        }

        try:
            # Submit to CLOB API
            url = f"{CLOB_API}/order"
            response = self.session.post(url, json=payload, timeout=5)

            if response.status_code == 200:
                data = response.json()
                order_id = data.get("orderID")

                order.order_id = order_id
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now(timezone.utc)

                # Track order
                self.orders[order_id] = order
                self.active_orders.append(order_id)

                print(f"  ✓ Order submitted: {order_id[:8]}... {order.side.value} {order.size} @ {order.price:.3f}")
                return order_id

            else:
                print(f"  ✗ Order submission failed: {response.status_code} {response.text}")
                order.status = OrderStatus.REJECTED
                return None

        except Exception as e:
            print(f"  ✗ Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled successfully
        """
        if self.dry_run:
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                if order_id in self.active_orders:
                    self.active_orders.remove(order_id)
                print(f"  [DRY RUN] Would cancel order: {order_id[:8]}...")
                return True
            return False

        try:
            url = f"{CLOB_API}/order/{order_id}"
            response = self.session.delete(url, timeout=5)

            if response.status_code == 200:
                if order_id in self.orders:
                    self.orders[order_id].status = OrderStatus.CANCELLED
                    if order_id in self.active_orders:
                        self.active_orders.remove(order_id)
                print(f"  ✓ Order cancelled: {order_id[:8]}...")
                return True
            else:
                print(f"  ✗ Cancel failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ✗ Cancel error: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Query order status from CLOB API.

        Args:
            order_id: Order ID to query

        Returns:
            Order status dict if found
        """
        if self.dry_run:
            order = self.orders.get(order_id)
            if order:
                return {
                    "orderID": order_id,
                    "status": order.status.value,
                    "size": order.size,
                    "filledSize": order.filled_size,
                }
            return None

        try:
            url = f"{CLOB_API}/order/{order_id}"
            response = self.session.get(url, timeout=5)

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            print(f"  ✗ Status query error: {e}")
            return None

    def update_order_status(self, order_id: str):
        """
        Update local order status from CLOB API.

        Fetches latest status and fill information.
        """
        status_data = self.get_order_status(order_id)

        if not status_data or order_id not in self.orders:
            return

        order = self.orders[order_id]

        # Update status
        api_status = status_data.get("status", "").lower()
        if api_status == "live":
            order.status = OrderStatus.OPEN
        elif api_status == "matched":
            order.status = OrderStatus.MATCHED
        elif api_status == "filled":
            order.status = OrderStatus.FILLED
            if not order.filled_at:
                order.filled_at = datetime.now(timezone.utc)
        elif api_status == "cancelled":
            order.status = OrderStatus.CANCELLED

        # Update fills
        filled_size = float(status_data.get("filledSize", 0))
        if filled_size > order.filled_size:
            order.filled_size = filled_size

            # Check if partially filled or fully filled
            if filled_size >= order.size:
                order.status = OrderStatus.FILLED
                if not order.filled_at:
                    order.filled_at = datetime.now(timezone.utc)
            elif filled_size > 0:
                order.status = OrderStatus.PARTIALLY_FILLED

        # Remove from active if no longer active
        if not order.is_active and order_id in self.active_orders:
            self.active_orders.remove(order_id)

    def update_all_orders(self):
        """Update status for all active orders."""
        for order_id in list(self.active_orders):  # Copy list to allow modification
            self.update_order_status(order_id)

    def get_balance(self, token_address: Optional[str] = None) -> float:
        """
        Query account balance.

        Args:
            token_address: Token to query (default: USDC)

        Returns:
            Balance in human-readable units
        """
        if not self.address:
            return 0.0

        # TODO: Implement actual balance query from Polygon
        # For now, return a placeholder
        print("  ⚠ Balance query not implemented - returning 0")
        return 0.0

    def close(self):
        """Clean up resources."""
        self.session.close()


if __name__ == "__main__":
    # Example usage (dry run mode)
    print("=" * 60)
    print("Polymarket Executor - Example Usage")
    print("=" * 60)

    # Initialize in dry run mode (safe for testing)
    executor = PolymarketExecutor(dry_run=True)

    # Create a test order
    order = executor.create_order(
        token_id="0x1234...",  # Example token ID
        side=OrderSide.BUY,
        price=0.45,
        size=10.0
    )

    # Submit (will be logged but not sent in dry run mode)
    order_id = executor.submit_order(order)

    # Check status
    if order_id:
        executor.update_order_status(order_id)
        status = executor.orders[order_id].status
        print(f"\nOrder status: {status.value}")

    # Cancel
    if order_id:
        executor.cancel_order(order_id)

    executor.close()
