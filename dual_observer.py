#!/usr/bin/env python3
"""
Dual observer: Monitor Polymarket + Kalshi to measure timing advantage.

Watches both platforms simultaneously to detect price propagation delays.
Legal for U.S. users: Observes Polymarket (public data), trades on Kalshi.

Key insight: Polymarket prices often lead Kalshi by 5-30 seconds.
This creates arbitrage opportunities:
  1. See price move on Polymarket
  2. Act on Kalshi before it catches up
  3. Profit from timing advantage

Usage:
    python dual_observer.py \
        --kalshi-email YOUR_EMAIL \
        --kalshi-password YOUR_PASSWORD \
        --market-pairs BTC ETH \
        --s3-bucket my-data
"""
import argparse
import asyncio
import json
import gzip
import sys
import time
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

# Polymarket public API (no authentication needed)
import requests

# Kalshi client
sys.path.insert(0, ".")
from helpers.kalshi_client import Kalshi

Client


@dataclass
class PriceEvent:
    """Price change event."""
    platform: str  # "polymarket" or "kalshi"
    market_id: str
    timestamp: datetime
    yes_price: float
    no_price: float
    spread: float
    volume_24h: float = 0.0


@dataclass
class TimingLag:
    """Measured timing lag between platforms."""
    market_pair: str
    polymarket_event: PriceEvent
    kalshi_event: PriceEvent
    lag_ms: float  # Milliseconds Kalshi lagged behind Polymarket
    price_delta: float  # How much price changed
    direction: str  # "up" or "down"


class DualObserver:
    """
    Observes both Polymarket and Kalshi to measure timing advantage.

    Records price movements and calculates how long Kalshi takes to catch up
    to Polymarket price changes.
    """

    def __init__(
        self,
        kalshi_email: str,
        kalshi_password: str,
        market_pairs: Dict[str, Tuple[str, str]],  # {pair_name: (poly_id, kalshi_ticker)}
        s3_bucket: Optional[str] = None,
        local_dir: str = "data/timing",
        tick_interval: float = 0.5,
    ):
        """
        Initialize dual observer.

        Args:
            kalshi_email: Kalshi account email
            kalshi_password: Kalshi account password
            market_pairs: Dict mapping pair names to (Polymarket ID, Kalshi ticker)
                Example: {"BTC100K": ("0x123...", "KXBTC-23DEC31-T99999")}
            s3_bucket: S3 bucket for data upload (optional)
            local_dir: Local data directory
            tick_interval: Observation interval in seconds
        """
        self.market_pairs = market_pairs
        self.s3_bucket = s3_bucket
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.tick_interval = tick_interval

        # Initialize Kalshi client
        print("🔌 Connecting to Kalshi...")
        self.kalshi = KalshiClient(kalshi_email, kalshi_password)

        # Polymarket public API (no auth needed)
        self.polymarket_base_url = "https://clob.polymarket.com"

        # State tracking
        self.last_prices: Dict[str, Dict[str, PriceEvent]] = {
            pair: {"polymarket": None, "kalshi": None}
            for pair in market_pairs
        }

        # Timing lag measurements
        self.timing_lags: List[TimingLag] = []
        self.lag_buffer_size = 1000  # Write every 1000 measurements

        # Statistics
        self.iteration = 0
        self.start_time = None
        self.running = False

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"✓ Dual observer initialized")
        print(f"   Watching {len(market_pairs)} market pairs")
        print(f"   Tick interval: {tick_interval}s")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n\n⏸️  Received signal {signum}, shutting down...")
        self.running = False

    def get_polymarket_price(self, condition_id: str) -> Optional[PriceEvent]:
        """
        Get current price from Polymarket (public API).

        No authentication required - this is legal to observe.
        """
        try:
            # Get orderbook via public API
            response = requests.get(
                f"{self.polymarket_base_url}/book",
                params={"market": condition_id},
                timeout=2
            )

            if response.status_code != 200:
                return None

            book = response.json()

            # Extract best bid/ask
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if not bids or not asks:
                return None

            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])

            yes_price = (best_bid + best_ask) / 2
            no_price = 1 - yes_price
            spread = best_ask - best_bid

            return PriceEvent(
                platform="polymarket",
                market_id=condition_id,
                timestamp=datetime.now(timezone.utc),
                yes_price=yes_price,
                no_price=no_price,
                spread=spread,
            )

        except Exception as e:
            print(f"⚠️  Polymarket fetch failed: {e}")
            return None

    def get_kalshi_price(self, ticker: str) -> Optional[PriceEvent]:
        """Get current price from Kalshi."""
        try:
            orderbook = self.kalshi.get_orderbook(ticker, depth=1)

            if not orderbook["yes"] or not orderbook["no"]:
                return None

            # Kalshi prices are in cents
            yes_bid = orderbook["yes"][0][0] / 100
            no_bid = orderbook["no"][0][0] / 100

            # Estimate mid price (Kalshi doesn't have asks in same format)
            yes_price = yes_bid
            no_price = no_bid
            spread = abs(yes_price - no_price)

            return PriceEvent(
                platform="kalshi",
                market_id=ticker,
                timestamp=datetime.now(timezone.utc),
                yes_price=yes_price,
                no_price=no_price,
                spread=spread,
            )

        except Exception as e:
            print(f"⚠️  Kalshi fetch failed: {e}")
            return None

    def detect_timing_lag(
        self,
        pair_name: str,
        poly_event: PriceEvent,
        kalshi_event: PriceEvent
    ) -> Optional[TimingLag]:
        """
        Detect if Polymarket price movement preceded Kalshi.

        Looks for:
        1. Significant price change on Polymarket
        2. Similar price change on Kalshi shortly after
        3. Measures the delay
        """
        # Get previous prices
        prev_poly = self.last_prices[pair_name]["polymarket"]
        prev_kalshi = self.last_prices[pair_name]["kalshi"]

        if not prev_poly or not prev_kalshi:
            return None  # Need history

        # Check if Polymarket price changed significantly
        poly_delta = poly_event.yes_price - prev_poly.yes_price
        threshold = 0.02  # 2% price change

        if abs(poly_delta) < threshold:
            return None  # No significant move

        # Check if Kalshi followed
        kalshi_delta = kalshi_event.yes_price - prev_kalshi.yes_price

        # Same direction?
        if poly_delta * kalshi_delta < 0:
            return None  # Opposite directions

        # Calculate time lag
        lag_ms = (kalshi_event.timestamp - poly_event.timestamp).total_seconds() * 1000

        # Polymarket should lead (negative lag)
        if lag_ms > 0:
            return None  # Kalshi moved first (unusual)

        # Record timing lag
        return TimingLag(
            market_pair=pair_name,
            polymarket_event=poly_event,
            kalshi_event=kalshi_event,
            lag_ms=abs(lag_ms),
            price_delta=poly_delta,
            direction="up" if poly_delta > 0 else "down",
        )

    def record_timing_lag(self, lag: TimingLag):
        """Record timing lag measurement."""
        self.timing_lags.append(lag)

        # Print interesting lags
        if lag.lag_ms > 1000:  # >1 second lag
            print(f"⏱️  [{lag.market_pair}] Kalshi lagged by {lag.lag_ms:.0f}ms, "
                  f"price moved {lag.price_delta:+.3f} ({lag.direction})")

        # Write to disk periodically
        if len(self.timing_lags) >= self.lag_buffer_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        """Write timing measurements to disk."""
        if not self.timing_lags:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.local_dir / f"timing_lags_{timestamp}.json.gz"

        data = {
            "start_time": self.timing_lags[0].polymarket_event.timestamp.isoformat(),
            "end_time": self.timing_lags[-1].polymarket_event.timestamp.isoformat(),
            "count": len(self.timing_lags),
            "lags": [asdict(lag) for lag in self.timing_lags],
        }

        # Convert datetime objects to strings
        for lag in data["lags"]:
            lag["polymarket_event"]["timestamp"] = lag["polymarket_event"]["timestamp"].isoformat()
            lag["kalshi_event"]["timestamp"] = lag["kalshi_event"]["timestamp"].isoformat()

        with gzip.open(filename, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        # Statistics
        lags = [lag.lag_ms for lag in self.timing_lags]
        avg_lag = sum(lags) / len(lags)
        max_lag = max(lags)

        print(f"💾 Wrote {len(self.timing_lags)} timing measurements")
        print(f"   Avg lag: {avg_lag:.0f}ms, Max lag: {max_lag:.0f}ms")

        # Upload to S3 if configured
        if self.s3_bucket:
            self.upload_to_s3(filename)

        # Clear buffer
        self.timing_lags = []

    def upload_to_s3(self, filepath: Path):
        """Upload file to S3."""
        try:
            import boto3
            s3 = boto3.client("s3")

            date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
            s3_key = f"timing-data/{date_prefix}/{filepath.name}"

            s3.upload_file(str(filepath), self.s3_bucket, s3_key)
            print(f"☁️  Uploaded to s3://{self.s3_bucket}/{s3_key}")

        except ImportError:
            print(f"⚠️  boto3 not installed, skipping S3 upload")
        except Exception as e:
            print(f"⚠️  S3 upload failed: {e}")

    async def run(self):
        """Main observation loop."""
        print("\n" + "=" * 60)
        print("🚀 Starting Dual Observer")
        print("=" * 60)
        print(f"\nMonitoring {len(self.market_pairs)} market pairs:")
        for pair_name, (poly_id, kalshi_ticker) in self.market_pairs.items():
            print(f"  • {pair_name}: Polymarket ({poly_id[:8]}...) ↔ Kalshi ({kalshi_ticker})")

        print(f"\n🔄 Starting observation loop (tick={self.tick_interval}s)")
        print("   Press Ctrl+C to stop\n")

        self.running = True
        self.start_time = time.time()

        try:
            while self.running:
                loop_start = time.time()

                # Observe each market pair
                for pair_name, (poly_id, kalshi_ticker) in self.market_pairs.items():
                    # Get Polymarket price (public API)
                    poly_price = self.get_polymarket_price(poly_id)

                    # Get Kalshi price
                    kalshi_price = self.get_kalshi_price(kalshi_ticker)

                    if poly_price and kalshi_price:
                        # Check for timing lag
                        lag = self.detect_timing_lag(pair_name, poly_price, kalshi_price)
                        if lag:
                            self.record_timing_lag(lag)

                        # Update last prices
                        self.last_prices[pair_name]["polymarket"] = poly_price
                        self.last_prices[pair_name]["kalshi"] = kalshi_price

                self.iteration += 1

                # Status update every 100 iterations
                if self.iteration % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.iteration / elapsed
                    print(f"✓ Iteration {self.iteration} | "
                          f"{len(self.timing_lags)} lags buffered | "
                          f"{rate:.1f} ticks/sec")

                # Sleep to maintain tick rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.tick_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("\n\n🛑 Shutting down...")
            self.flush_to_disk()
            print("✓ Dual observer stopped cleanly")


async def main():
    parser = argparse.ArgumentParser(
        description="Dual observer: Measure Polymarket→Kalshi timing advantage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor BTC and ETH markets
  python dual_observer.py \
      --kalshi-email your@email.com \
      --kalshi-password yourpassword \
      --market-pairs BTC ETH \
      --s3-bucket my-timing-data

  # Custom market pairs
  python dual_observer.py \
      --kalshi-email your@email.com \
      --kalshi-password yourpassword \
      --poly-id 0x123... \
      --kalshi-ticker KXBTC-23DEC31-T99999 \
      --pair-name BTC100K
        """
    )
    parser.add_argument("--kalshi-email", required=True, help="Kalshi account email")
    parser.add_argument("--kalshi-password", required=True, help="Kalshi account password")
    parser.add_argument("--market-pairs", nargs="+", help="Market keywords (BTC, ETH, etc.)")
    parser.add_argument("--poly-id", help="Polymarket condition ID (for single pair)")
    parser.add_argument("--kalshi-ticker", help="Kalshi ticker (for single pair)")
    parser.add_argument("--pair-name", help="Name for market pair (for single pair)")
    parser.add_argument("--s3-bucket", help="S3 bucket for data upload")
    parser.add_argument("--local-dir", default="data/timing", help="Local data directory")
    parser.add_argument("--tick", type=float, default=0.5, help="Tick interval (seconds)")

    args = parser.parse_args()

    # Build market pairs dict
    market_pairs = {}

    if args.poly_id and args.kalshi_ticker and args.pair_name:
        # Single pair specified
        market_pairs[args.pair_name] = (args.poly_id, args.kalshi_ticker)

    elif args.market_pairs:
        # Auto-discover pairs (placeholder - would need API calls to match markets)
        print("⚠️  Auto-discovery not yet implemented")
        print("   Please specify --poly-id, --kalshi-ticker, and --pair-name")
        sys.exit(1)

    else:
        print("❌ Must specify either:")
        print("   1. --market-pairs (auto-discovery, not yet implemented)")
        print("   2. --poly-id, --kalshi-ticker, and --pair-name (manual)")
        sys.exit(1)

    # Create and run observer
    observer = DualObserver(
        kalshi_email=args.kalshi_email,
        kalshi_password=args.kalshi_password,
        market_pairs=market_pairs,
        s3_bucket=args.s3_bucket,
        local_dir=args.local_dir,
        tick_interval=args.tick,
    )

    await observer.run()


if __name__ == "__main__":
    asyncio.run(main())
