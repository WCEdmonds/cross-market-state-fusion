#!/usr/bin/env python3
"""
Market data observer - collects state observations without trading.

Runs 24/7 on cheap AWS instance, uploads data to S3 periodically.

Usage:
    python observer.py --markets TRUMP2024 ETH10K --s3-bucket my-market-data
    python observer.py --markets all --s3-bucket my-market-data
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
from typing import Dict, List, Optional
from dataclasses import asdict

# Import existing helpers
sys.path.insert(0, ".")
from helpers import (
    get_15m_markets,
    BinanceStreamer,
    OrderbookStreamer,
    FuturesStreamer,
    Market,
    FuturesState,
)
from strategies import MarketState


class MarketObserver:
    """
    Observes market data and records states for offline training.

    Lightweight version of TradingEngine - only collects data, no trading.
    """

    def __init__(
        self,
        markets: List[str],
        s3_bucket: Optional[str] = None,
        local_dir: str = "data",
        tick_interval: float = 0.5,
        buffer_size: int = 1000,
    ):
        self.target_markets = markets
        self.s3_bucket = s3_bucket
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.tick_interval = tick_interval
        self.buffer_size = buffer_size

        # Initialize streamers
        print("🔌 Initializing market data streamers...")
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])

        # State
        self.markets: Dict[str, Market] = {}
        self.states: Dict[str, MarketState] = {}
        self.open_prices: Dict[str, float] = {}  # Binance price at market open

        # Recording buffer (per market)
        self.observations: Dict[str, List[dict]] = {}

        # Stats
        self.iteration = 0
        self.start_time = None
        self.running = False

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"✓ Observer initialized")
        print(f"   Markets: {markets}")
        print(f"   Local dir: {self.local_dir}")
        print(f"   S3 bucket: {s3_bucket or 'disabled'}")
        print(f"   Tick interval: {tick_interval}s")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n\n⏸️  Received signal {signum}, shutting down...")
        self.running = False

    async def fetch_markets(self):
        """Fetch available markets from Polymarket."""
        print("\n📊 Fetching active markets...")
        all_markets = await get_15m_markets()

        if not all_markets:
            print("❌ No active markets found")
            return

        # Filter to target markets if specified
        if "all" in self.target_markets:
            self.markets = {m.condition_id: m for m in all_markets}
        else:
            self.markets = {
                m.condition_id: m for m in all_markets
                if any(keyword.lower() in m.question.lower()
                       for keyword in self.target_markets)
            }

        if not self.markets:
            print(f"❌ No markets found matching: {self.target_markets}")
            return

        print(f"✓ Found {len(self.markets)} markets:")
        for cid, market in self.markets.items():
            print(f"   • {market.question[:60]}...")
            # Initialize observation buffer for this market
            self.observations[cid] = []

    def get_market_state(self, cid: str, market: Market) -> Optional[MarketState]:
        """
        Build MarketState for a given market.

        Same logic as TradingEngine._build_state() but without position tracking.
        """
        # Get orderbook data
        book = self.orderbook_streamer.books.get(cid)
        if not book:
            return None

        # Extract orderbook features
        best_bid = float(book.get("bids", [[0, 0]])[0][0]) if book.get("bids") else 0
        best_ask = float(book.get("asks", [[0, 0]])[0][0]) if book.get("asks") else 0
        spread = best_ask - best_bid if best_ask and best_bid else 0

        # Current probability (mid-market)
        prob = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.5

        # Time remaining (fraction of 15 min)
        now = datetime.now(timezone.utc)
        time_remaining = max(0, (market.end_date - now).total_seconds() / 900)  # 900s = 15min

        # Get Binance price
        asset = market.asset
        binance_price = self.price_streamer.prices.get(asset, 0)

        # Binance change since market open
        open_price = self.open_prices.get(cid, binance_price)
        if cid not in self.open_prices and binance_price > 0:
            self.open_prices[cid] = binance_price
            open_price = binance_price

        binance_change = (binance_price / open_price - 1) if open_price > 0 else 0

        # Orderbook imbalance
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        bid_vol_l1 = float(bids[0][1]) if bids else 0
        ask_vol_l1 = float(asks[0][1]) if asks else 0
        total_l1 = bid_vol_l1 + ask_vol_l1
        imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0

        # L5 imbalance
        bid_vol_l5 = sum(float(b[1]) for b in bids[:5])
        ask_vol_l5 = sum(float(a[1]) for a in asks[:5])
        total_l5 = bid_vol_l5 + ask_vol_l5
        imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0

        # Futures data
        futures_state = self.futures_streamer.states.get(asset)
        funding_rate = futures_state.funding_rate if futures_state else 0
        oi_change = futures_state.oi_change if futures_state else 0
        long_short_ratio = futures_state.long_short_ratio if futures_state else 1.0

        # Build state
        state = MarketState(
            asset=asset,
            prob=prob,
            time_remaining=time_remaining,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            order_book_imbalance_l1=imbalance_l1,
            order_book_imbalance_l5=imbalance_l5,
            binance_price=binance_price,
            binance_change=binance_change,
            # Note: No position tracking in observer mode
            has_position=False,
            position_side=None,
            position_pnl=0.0,
        )

        # Add to history for temporal features
        prev_state = self.states.get(cid)
        if prev_state:
            state.prob_history = prev_state.prob_history[-4:] + [prev_state.prob]

        return state

    def record_observation(self, cid: str, market: Market, state: MarketState):
        """Record observation with metadata."""
        obs = {
            "timestamp": datetime.utcnow().isoformat(),
            "cid": cid,
            "market": {
                "question": market.question,
                "asset": market.asset,
                "end_date": market.end_date.isoformat(),
            },
            "state": {
                # Core features
                "prob": state.prob,
                "time_remaining": state.time_remaining,
                # Orderbook
                "best_bid": state.best_bid,
                "best_ask": state.best_ask,
                "spread": state.spread,
                "imbalance_l1": state.order_book_imbalance_l1,
                "imbalance_l5": state.order_book_imbalance_l5,
                # Binance
                "binance_price": state.binance_price,
                "binance_change": state.binance_change,
                # History
                "prob_history": state.prob_history,
            }
        }

        self.observations[cid].append(obs)

        # Check if we need to flush this market's buffer
        if len(self.observations[cid]) >= self.buffer_size:
            self.flush_to_disk(cid)

    def flush_to_disk(self, cid: str):
        """Write observations for a market to local disk (compressed JSON)."""
        if not self.observations.get(cid):
            return

        # Create market-specific directory
        market_dir = self.local_dir / cid
        market_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = market_dir / f"obs_{timestamp}.json.gz"

        data = {
            "cid": cid,
            "start_time": self.observations[cid][0]["timestamp"],
            "end_time": self.observations[cid][-1]["timestamp"],
            "count": len(self.observations[cid]),
            "observations": self.observations[cid],
        }

        with gzip.open(filename, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        print(f"💾 [{cid[:8]}] Wrote {len(self.observations[cid])} observations to {filename.name}")

        # Upload to S3 if configured
        if self.s3_bucket:
            self.upload_to_s3(filename, cid)

        # Clear buffer
        self.observations[cid] = []

    def upload_to_s3(self, filepath: Path, cid: str):
        """Upload file to S3."""
        try:
            import boto3
            s3 = boto3.client("s3")

            # Key format: market-data/{CID}/{date}/{filename}
            date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
            s3_key = f"market-data/{cid}/{date_prefix}/{filepath.name}"

            s3.upload_file(str(filepath), self.s3_bucket, s3_key)
            print(f"☁️  [{cid[:8]}] Uploaded to s3://{self.s3_bucket}/{s3_key}")

        except ImportError:
            print(f"⚠️  boto3 not installed, skipping S3 upload")
        except Exception as e:
            print(f"⚠️  S3 upload failed: {e}")

    def flush_all(self):
        """Flush all market buffers to disk."""
        print("\n💾 Flushing all buffers...")
        for cid in list(self.observations.keys()):
            if self.observations[cid]:
                self.flush_to_disk(cid)
        print("✓ All data saved")

    async def run(self):
        """Main observation loop."""
        print("\n" + "=" * 60)
        print("🚀 Starting Market Observer")
        print("=" * 60)

        # Fetch markets
        await self.fetch_markets()

        if not self.markets:
            print("❌ No markets to observe")
            return

        # Start streamers
        print("\n📡 Starting data streamers...")
        await self.price_streamer.start()
        await self.futures_streamer.start()

        # Subscribe to orderbooks
        for cid in self.markets.keys():
            await self.orderbook_streamer.subscribe(cid)

        print("✓ Streamers started")

        # Wait for initial data
        print("\n⏳ Waiting for initial market data...")
        await asyncio.sleep(2)

        # Main loop
        print(f"\n🔄 Starting observation loop (tick={self.tick_interval}s)")
        print("   Press Ctrl+C to stop\n")

        self.running = True
        self.start_time = time.time()

        try:
            while self.running:
                loop_start = time.time()

                # Process each market
                for cid, market in list(self.markets.items()):
                    # Check if market is still active
                    if datetime.now(timezone.utc) > market.end_date:
                        print(f"⏰ [{cid[:8]}] Market expired, removing")
                        self.flush_to_disk(cid)
                        del self.markets[cid]
                        continue

                    # Build state
                    state = self.get_market_state(cid, market)

                    if state:
                        # Record observation
                        self.record_observation(cid, market, state)

                        # Update stored state
                        self.states[cid] = state

                self.iteration += 1

                # Status update every 100 iterations
                if self.iteration % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.iteration / elapsed
                    total_obs = sum(len(obs) for obs in self.observations.values())
                    print(f"✓ Iteration {self.iteration} | "
                          f"{len(self.markets)} markets | "
                          f"{total_obs} buffered | "
                          f"{rate:.1f} ticks/sec")

                # Refresh markets every 100 iterations
                if self.iteration % 100 == 0:
                    await self.fetch_markets()

                # Sleep to maintain tick rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.tick_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            print("\n\n🛑 Shutting down...")
            self.flush_all()

            # Stop streamers
            await self.price_streamer.stop()
            await self.futures_streamer.stop()
            await self.orderbook_streamer.close()

            print("✓ Observer stopped cleanly")


async def main():
    parser = argparse.ArgumentParser(
        description="Market data observer for offline training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Observe specific markets
  python observer.py --markets TRUMP2024 ETH10K --s3-bucket my-data

  # Observe all available markets
  python observer.py --markets all --s3-bucket my-data

  # Local only (no S3)
  python observer.py --markets all --local-dir ./data

  # Custom tick rate
  python observer.py --markets all --tick 1.0 --s3-bucket my-data
        """
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        required=True,
        help='Market keywords to observe (or "all" for all markets)',
    )
    parser.add_argument(
        "--s3-bucket",
        help="S3 bucket for data upload (optional)",
    )
    parser.add_argument(
        "--local-dir",
        default="data",
        help="Local data directory (default: data)",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=0.5,
        help="Tick interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Number of observations before writing to disk (default: 1000)",
    )

    args = parser.parse_args()

    # Validate S3 bucket if provided
    if args.s3_bucket:
        try:
            import boto3
            s3 = boto3.client("s3")
            # Test access
            s3.head_bucket(Bucket=args.s3_bucket)
            print(f"✓ S3 bucket '{args.s3_bucket}' accessible")
        except ImportError:
            print("❌ boto3 not installed. Install: pip install boto3")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  S3 bucket check failed: {e}")
            print("   Continuing anyway (will attempt uploads later)")

    # Create and run observer
    observer = MarketObserver(
        markets=args.markets,
        s3_bucket=args.s3_bucket,
        local_dir=args.local_dir,
        tick_interval=args.tick,
        buffer_size=args.buffer_size,
    )

    await observer.run()


if __name__ == "__main__":
    asyncio.run(main())
