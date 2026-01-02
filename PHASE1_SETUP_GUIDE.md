# Phase 1: Live Trading Execution - Setup Guide

**Status**: ✅ Implementation Complete
**Risk Level**: ⚠️ HIGH - Real money at risk
**Recommended Start Size**: $5-10 per trade

---

## What Was Built

Phase 1 implements the execution infrastructure needed to transition from paper trading to live trading:

### Core Components

1. **`helpers/polymarket_executor.py`** - Order execution layer
   - EIP-712 signature generation for Polymarket orders
   - Order placement via CLOB REST API
   - Order status tracking and updates
   - Cancel orders functionality
   - Connection pooling for latency optimization

2. **`helpers/order_manager.py`** - Order lifecycle manager
   - Track order state transitions (pending → open → filled)
   - Aggregate fills into positions
   - Calculate realized P&L from actual execution prices
   - Position reconciliation (paper vs actual)
   - Slippage tracking

3. **Modified `run.py`** - Integrated live trading support
   - `--live` flag to enable real trading
   - `--private-key` for order signing
   - Dual execution paths (paper vs live)
   - Order status polling (every 1 second)
   - Position reconciliation (every 10 seconds)
   - Safety confirmations before live trading

---

## Installation

### 1. Install Dependencies

```bash
# Core dependencies (already installed)
pip install mlx websockets flask flask-socketio numpy requests

# New: Live trading dependencies
pip install eth-account web3
```

### 2. Verify Installation

```bash
python -c "from eth_account import Account; print('✓ eth-account installed')"
python -c "from web3 import Web3; print('✓ web3 installed')"
```

---

## Configuration

### 1. Get a Private Key

**Option A: Create New Wallet**
```python
from eth_account import Account
import secrets

# Generate new private key
private_key = "0x" + secrets.token_hex(32)
account = Account.from_key(private_key)

print(f"Private Key: {private_key}")
print(f"Address: {account.address}")
```

**Option B: Export from MetaMask**
- Open MetaMask
- Click account menu → Account Details → Export Private Key
- Enter password
- Copy private key (starts with 0x)

**SECURITY WARNING**:
- ⚠️ Never commit private keys to git
- ⚠️ Never share private keys
- ⚠️ Use a dedicated wallet for trading (not your main wallet)
- ⚠️ Fund with only the amount you're willing to lose

### 2. Fund Your Wallet

Transfer USDC to your trading wallet address on Polygon:
- Minimum: $50 USDC (for $5 trades with buffer)
- Recommended: $100-200 USDC for initial testing

**How to get USDC on Polygon**:
1. Buy USDC on Coinbase/Binance
2. Bridge to Polygon using:
   - Polygon Bridge: https://portal.polygon.technology/
   - Multichain: https://multichain.org/
   - Hop Protocol: https://hop.exchange/

### 3. Set Environment Variable (Recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc
export POLYMARKET_KEY="0x1234567890abcdef..."  # Your private key

# Load in current shell
source ~/.bashrc
```

---

## Usage

### Paper Trading (Default)

```bash
# Standard paper trading - no changes
python run.py rl --train --size 50
```

### Live Trading (Real Money)

**CRITICAL: The executor defaults to `dry_run=True` for safety.**

To enable REAL orders, you must modify the executor initialization in `run.py`:

```python
# Find this line in run.py (around line 132):
self.executor = PolymarketExecutor(
    private_key=private_key,
    dry_run=True  # CHANGE THIS TO False FOR REAL ORDERS
)
```

**Then run**:
```bash
# With environment variable
python run.py rl --live --private-key $POLYMARKET_KEY --size 5

# Or directly (NOT RECOMMENDED - visible in process list)
python run.py rl --live --private-key "0x123..." --size 5
```

**Confirmation Prompt**:
```
⚠️  WARNING: LIVE TRADING MODE REQUESTED ⚠️
This will place REAL orders with REAL money on Polymarket.
Trade size: $5 per order
Max exposure: $20 (4 concurrent markets)

Type 'I UNDERSTAND' to proceed:
```

Type exactly `I UNDERSTAND` and press Enter.

### Testing Dry Run Mode

Even with `--live` flag, the executor starts in `dry_run=True` by default:

```bash
python run.py rl --live --private-key $POLYMARKET_KEY --size 5
```

You'll see:
```
✓ Executor initialized (DRY RUN mode)
  To enable real orders, modify executor.dry_run = False
```

Orders will be logged but **NOT** submitted to Polymarket.

---

## How It Works

### Execution Flow (Live Mode)

```
1. Strategy generates Action (BUY/SELL/HOLD)
   ↓
2. TradingEngine.execute_action() routes to _execute_live()
   ↓
3. Determine token ID and price:
   - BUY → UP token at current prob
   - SELL → DOWN token at (1 - prob)
   ↓
4. PolymarketExecutor.create_order()
   - Build order struct
   - Generate EIP-712 signature
   ↓
5. PolymarketExecutor.submit_order()
   - POST to Polymarket CLOB API
   - Receive order_id
   ↓
6. OrderManager.register_order()
   - Track order lifecycle
   ↓
7. Decision loop polls every 1 second:
   - executor.update_all_orders()
   - order_manager.update_order()
   - Reconcile positions every 10 seconds
   ↓
8. When filled:
   - order_manager handles fill
   - Track slippage vs expected price
   - Calculate realized P&L
```

### Order Status Tracking

Orders go through these states:
```
PENDING → SUBMITTED → OPEN → MATCHED → FILLED
                        ↓
                    CANCELLED / REJECTED
```

### Position Reconciliation

Every 10 seconds, the system reconciles:
- **Paper position**: Expected position from strategy decisions
- **Actual position**: Real position from order fills

If discrepancy detected (size diff >$1 or price diff >0.05):
```
⚠ POSITION DISCREPANCY for condition_abc123:
  Paper: $50.00 @ 0.450
  Actual: $48.50 @ 0.462
  Diff: $1.50 size, 0.012 price
```

This could indicate:
- Partial fills
- Slippage
- Order rejection
- API errors

---

## Safety Features

### 1. Multi-Layer Confirmation

```
--live flag → Private key check → Confirmation prompt → dry_run flag
```

All 4 must be passed to enable real trading.

### 2. Default to Dry Run

Even with `--live`, executor defaults to `dry_run=True`.
You must manually edit code to set `dry_run=False`.

### 3. Order Size Limits

Start with tiny sizes:
- Initial testing: $5-10 per trade
- After 50+ trades: $25-50
- After 200+ trades: $100-250
- After proven: $500 max (per market)

### 4. Position Reconciliation

Automatic checks every 10 seconds detect:
- Execution failures
- Partial fills
- Unexpected position state

### 5. Execution Statistics

After each session:
```
ORDER EXECUTION STATS
Total orders: 45
Total fills: 42 (93.3% fill rate)
Avg fill latency: 1.23s
Avg slippage: +0.0023
Realized P&L: $+12.50
Active positions: 2
```

---

## Monitoring

### Console Output

**Order Submission**:
```
  ✓ Order signed: BUY 10.0 @ 0.450
  ✓ Order submitted: a1b2c3d4... BUY 10.0 @ 0.450
  📋 Registered order: a1b2c3d4... (BTC)
    OPEN BTC UP (MD) $10 @ 0.450
```

**Order Fill**:
```
  ✓ Order filled in 1.23s: a1b2c3d4...
  💸 Slippage: +0.0012 (+0.27%)
  📊 Fill: +10.00 @ 0.451 | Total: 22.17 shares @ 0.450 avg
```

**Position Close**:
```
  💰 Position closed: +2.50 PnL | Total PnL: $+12.50
```

### Dashboard (Optional)

If running with `--dashboard`:
```bash
python run.py rl --live --private-key $POLYMARKET_KEY --size 5 --dashboard
```

Open http://localhost:5050 to see:
- Real-time order status
- Fill tracking
- Slippage metrics
- P&L by asset

---

## Testing Checklist

### Before Going Live

- [ ] Install `eth-account` and `web3`
- [ ] Generate or export private key
- [ ] Fund wallet with $50-100 USDC on Polygon
- [ ] Set `POLYMARKET_KEY` environment variable
- [ ] Test dry run mode: `python run.py rl --live --private-key $POLYMARKET_KEY --size 5`
- [ ] Verify orders logged but not submitted
- [ ] Review Polymarket API documentation (update EIP-712 schema if needed)

### Initial Live Testing ($5-10 per trade)

- [ ] Modify `run.py` to set `executor.dry_run = False`
- [ ] Run with tiny size: `--size 5`
- [ ] Manually review each trade decision
- [ ] Verify orders appear on Polymarket UI
- [ ] Check fills match expected prices (within 1-2%)
- [ ] Confirm P&L calculation accurate
- [ ] Monitor for 50+ trades (3-5 hours)
- [ ] Analyze slippage distribution

### Scale-Up Testing ($25-50 per trade)

- [ ] Positive P&L over 100+ trades at $5-10 size
- [ ] Avg slippage <0.5%
- [ ] Fill rate >80%
- [ ] No unexpected position discrepancies
- [ ] Run for 200+ trades at $25-50 size
- [ ] Monitor market impact (prices moving against you)

### Production Trading ($100-500 per trade)

- [ ] Proven profitable over 500+ trades
- [ ] Sharpe ratio >1.0 (annualized)
- [ ] Max drawdown <15%
- [ ] Slippage remains acceptable at larger sizes
- [ ] Deploy with monitoring and alerts
- [ ] Consider cloud hosting (AWS us-east-1) for latency

---

## Important Notes

### 1. EIP-712 Signature Schema

**⚠️ CRITICAL**: The EIP-712 schema in `polymarket_executor.py` is a **PLACEHOLDER**.

You MUST update it with the actual Polymarket CLOB schema before going live:
- Check Polymarket documentation: https://docs.polymarket.com/
- Or inspect orders from Polymarket UI (browser dev tools)
- Look for: domain separator, type hash, order struct fields

**Current placeholder** (lines 68-112 in `polymarket_executor.py`):
```python
domain = {
    "name": "Polymarket CTF Exchange",  # UPDATE THIS
    "version": "1",                      # UPDATE THIS
    "chainId": self.chain_id,
    "verifyingContract": "0x4bFb..."    # UPDATE THIS
}
```

### 2. Order Placement API

The order placement endpoint and payload format are approximations:
```python
url = f"{CLOB_API}/order"  # Verify correct endpoint
payload = {
    "tokenID": ...,  # Verify field names
    "price": ...,
    # etc.
}
```

Check Polymarket API docs for:
- Correct endpoint URL
- Request payload structure
- Response format
- Rate limits
- Authentication (if required)

### 3. Fill Tracking

Current implementation polls order status every 1 second.
Consider subscribing to WebSocket for faster fill notifications if available.

### 4. Latency Optimization

Current setup:
- Connection pooling: ✅ Enabled (requests.Session)
- Keep-alive: ✅ Enabled
- Async I/O: ❌ Not fully async (using requests, not aiohttp)

For better latency:
- Switch to `aiohttp` for async HTTP
- Deploy to AWS us-east-1 (near Polygon RPC)
- Use ONNX Runtime for faster inference

---

## Troubleshooting

### "eth_account not installed"

```bash
pip install eth-account web3
```

### "Private key required for signing orders"

Make sure you passed `--private-key`:
```bash
python run.py rl --live --private-key $POLYMARKET_KEY --size 5
```

### "Order submission failed: 400"

Likely issues:
1. Invalid EIP-712 signature (schema mismatch)
2. Incorrect API endpoint or payload format
3. Insufficient balance
4. Token ID not found

Check Polymarket API docs and update `polymarket_executor.py`.

### Orders logged but not appearing on Polymarket

You're in `dry_run=True` mode. Change to `dry_run=False` in `run.py`:
```python
self.executor = PolymarketExecutor(
    private_key=private_key,
    dry_run=False  # ENABLE REAL ORDERS
)
```

### Position discrepancy warnings

Possible causes:
- Partial fills (order not fully filled)
- Slippage higher than expected
- Order rejected after paper position updated

Check order manager stats:
```python
self.order_manager.print_stats()
```

### High slippage

Reduce order size or use limit orders:
```python
order = self.executor.create_order(
    ...,
    post_only=True  # Maker orders only (no market taking)
)
```

---

## Next Steps

### Phase 2: Slippage Modeling

- [ ] Implement realistic fill simulation
- [ ] Model orderbook walking
- [ ] Calibrate slippage parameters from live data
- [ ] Compare paper vs live performance

### Phase 3: Latency Optimization

- [ ] Add latency instrumentation
- [ ] Measure P50/P95/P99 for each stage
- [ ] Deploy to AWS us-east-1
- [ ] Convert MLX → PyTorch/ONNX for cloud

### Phase 4: Risk Management

- [ ] Position limits
- [ ] Drawdown protection
- [ ] Kill switches (manual + automatic)
- [ ] Alerting (email/SMS)

### Phase 5: Extended Validation

- [ ] Paper trading with slippage (2 weeks)
- [ ] Tiny size live ($5-10, 2 weeks)
- [ ] Gradual scale-up
- [ ] Out-of-sample testing

---

## Support

**Questions/Issues**: Open an issue on GitHub
**Security Issues**: Never post private keys or wallet addresses publicly

**Remember**:
- Start tiny ($5-10)
- Test thoroughly
- Monitor actively
- Scale gradually
- Never risk more than you can afford to lose

Good luck! 🚀
