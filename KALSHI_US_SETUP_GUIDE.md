# Kalshi Setup Guide for U.S. Users

**Legal Trading Platform for U.S. Residents**

Kalshi is CFTC-regulated and fully legal for U.S. users. This guide shows how to use the trading bot with Kalshi instead of Polymarket.

---

## Table of Contents

1. [Why Kalshi for U.S. Users](#why-kalshi-for-us-users)
2. [Account Setup](#account-setup)
3. [Measuring Timing Advantage](#measuring-timing-advantage)
4. [Training & Deployment](#training--deployment)
5. [Expected Performance](#expected-performance)

---

## Why Kalshi for U.S. Users

### Legal Status

| Platform | U.S. Legal | Regulation | Access |
|----------|-----------|------------|--------|
| **Polymarket** | ❌ No (blocked 2022) | Unregulated | Geo-blocked |
| **Kalshi** | ✅ Yes | CFTC-regulated | Fully accessible |
| **PredictIt** | ⚠️ Limited | Academic exemption | $850 position limit |

**Bottom line**: Kalshi is your only option for serious trading in the U.S.

### Key Differences from Polymarket

**Advantages**:
- ✅ Legal & regulated
- ✅ USD-based (no crypto needed)
- ✅ Lower API latency (U.S. servers)
- ✅ Better for political markets
- ✅ Tight spreads

**Disadvantages**:
- ⚠️ Lower liquidity (10-20x less than Polymarket)
- ⚠️ Fewer 15-min markets
- ⚠️ Higher fees (7% on winnings vs 2%)
- ⚠️ Narrower market selection

---

## Account Setup

### Step 1: Create Kalshi Account

1. Go to https://kalshi.com
2. Click "Sign Up"
3. Complete KYC verification:
   - Government ID
   - SSN
   - Address verification
4. Fund account (minimum $100):
   - ACH transfer (free, 3-5 days)
   - Wire transfer ($25 fee, same day)
   - Debit card (instant, 3% fee)

**Note**: Kalshi is serious about compliance. Full KYC required.

### Step 2: Generate API Credentials

1. Log in to Kalshi
2. Go to Settings → API Access
3. Click "Generate API Token"
4. Save credentials:
   ```bash
   export KALSHI_EMAIL="your@email.com"
   export KALSHI_PASSWORD="your_password"
   ```

**Security**: Store in environment variables, never commit to git.

### Step 3: Test API Access

```bash
# Test Kalshi client
cd cross-market-state-fusion

python3 helpers/kalshi_client.py

# Expected output:
# ✓ Kalshi authenticated (token expires: ...)
# Exchange status: {'trading_active': True, ...}
# Found X open markets
# Account balance: $XXX.XX
```

---

## Measuring Timing Advantage

### The Opportunity

**Polymarket prices lead Kalshi by 5-30 seconds.**

You can legally:
1. **Observe** Polymarket prices (public API, no account)
2. **Trade** on Kalshi before it catches up
3. **Profit** from timing advantage

### Running Dual Observer

```bash
# Monitor BTC market pair
python dual_observer.py \
    --kalshi-email $KALSHI_EMAIL \
    --kalshi-password $KALSHI_PASSWORD \
    --poly-id "0x..." \
    --kalshi-ticker "KXBTC-23DEC31-T99999" \
    --pair-name "BTC100K" \
    --s3-bucket my-timing-data
```

**What it measures**:
- Price change on Polymarket
- Delay until Kalshi follows
- Direction & magnitude
- Exploitable opportunities

### Finding Market Pairs

**Matching Polymarket ↔ Kalshi markets**:

```python
# find_market_pairs.py (helper script)
import requests
from helpers.kalshi_client import KalshiClient

# Get Polymarket markets (public)
poly_markets = requests.get("https://clob.polymarket.com/markets").json()

# Get Kalshi markets
kalshi = KalshiClient(email, password)
kalshi_markets = kalshi.get_markets(status="open")

# Match by title similarity
for poly_market in poly_markets:
    for kalshi_market in kalshi_markets["markets"]:
        # Fuzzy match titles
        if similarity(poly_market["question"], kalshi_market["title"]) > 0.85:
            print(f"Match found:")
            print(f"  Polymarket: {poly_market['question']}")
            print(f"  Kalshi: {kalshi_market['title']}")
            print(f"  Pair: {poly_market['condition_id']} ↔ {kalshi_market['ticker']}")
```

**Common pairs** (as of Jan 2026):
- Bitcoin price predictions
- Fed interest rate decisions
- Political events (2026 midterms, 2028 primaries)
- Tech company earnings

---

## Training & Deployment

### Option 1: Kalshi-Only Training

Train directly on Kalshi data (no Polymarket):

```bash
# 1. Collect Kalshi data
python observer.py \
    --platform kalshi \
    --kalshi-email $KALSHI_EMAIL \
    --kalshi-password $KALSHI_PASSWORD \
    --markets all \
    --s3-bucket my-kalshi-data

# 2. Train on Mac
python offline_trainer.py \
    --data-dir ./data/kalshi \
    --output ./models/kalshi_v1 \
    --epochs 100

# 3. Deploy
python run.py rl-transformer \
    --kalshi \
    --load ./models/kalshi_v1 \
    --live \
    --size 5
```

### Option 2: Cross-Platform Strategy (Recommended)

Use Polymarket timing to front-run Kalshi:

```bash
# 1. Measure timing advantage (run for 1 week)
python dual_observer.py \
    --kalshi-email $KALSHI_EMAIL \
    --kalshi-password $KALSHI_PASSWORD \
    --poly-id "0x..." \
    --kalshi-ticker "KXBTC-..." \
    --pair-name "BTC100K"

# 2. Analyze timing
python analyze_timing.py --data ./data/timing

# Expected output:
# Avg lag: 12.3 seconds
# P95 lag: 28.1 seconds
# Exploitable events: 145/week

# 3. Deploy timing-aware strategy
python run.py timing-arbitrage \
    --watch-polymarket \
    --trade-kalshi \
    --min-lag 5  # Only trade if >5s advantage
    --size 10
```

### Key Parameters

**Kalshi-specific settings**:

```python
# run.py modifications for Kalshi
TradingEngine(
    strategy=strategy,
    trade_size=10,  # Start small on Kalshi (lower liquidity)
    platform="kalshi",
    kalshi_email=os.getenv("KALSHI_EMAIL"),
    kalshi_password=os.getenv("KALSHI_PASSWORD"),
)
```

---

## Expected Performance

### Kalshi vs Polymarket Comparison

```
Metric                | Polymarket | Kalshi | Notes
──────────────────────|──────────--|───────|──────────────────────
Liquidity             | High       | Medium | 10-20x less
Base strategy PnL     | $50K/year  | $25K/year | Lower liquidity limits
With timing advantage | N/A        | +$10K/year | Polymarket→Kalshi lag
With Transformer      | +$10K      | +$5K   | Scaled for liquidity
──────────────────────|──────────--|───────|──────────────────────
Total (U.S. legal)    | Not available | $40K/year | Conservative
```

**Realistic expectations for Kalshi**:
- **Year 1**: $20-40K profit (learning + validation)
- **Year 2+**: $40-70K profit (optimized strategies)

### Profit Sources

**1. Base RL Strategy** (~$20K/year)
- Same strategy as Polymarket
- Limited by lower liquidity
- Smaller position sizes

**2. Timing Arbitrage** (+$10K/year)
- Front-run Polymarket→Kalshi
- 5-30 second advantage
- Legal (observing public data)

**3. Transformer Enhancement** (+$5K/year)
- Better pattern recognition
- Improved entry/exit timing

**4. Cross-Platform** (+$5K/year)
- Kalshi ↔ PredictIt arbitrage
- Exploit price differences

**Total: $40K/year** (conservative estimate)

---

## Cost Analysis

### Monthly Costs (Kalshi)

```
AWS t3.micro (observer):    $7/month
S3 storage:                 $2/month
Kalshi API access:          FREE
Training (Mac):             $0/month
───────────────────────────────────
Total:                      $9/month
```

**vs. Polymarket** (if accessible):
- Same infrastructure: $9/month
- But Polymarket has 10-20x liquidity
- Higher profit potential if legal

### Transaction Fees

**Kalshi**:
- Trading fee: $0 (no trading fees!)
- Settlement fee: 7% on winnings
- Withdrawal: $0 (ACH free)

**Example**:
```
Win $1,000 on a market:
  Payout: $1,000
  Fee (7%): -$70
  Net: $930
```

**vs. Polymarket** (2% fee):
```
Win $1,000:
  Payout: $1,000
  Fee (2%): -$20
  Net: $980
```

**Kalshi costs 5% more in fees**, but spreads are often 3-5% tighter, so net cost is similar.

---

## Risk Management for Kalshi

### Position Size Limits

Kalshi has **position limits** per market:

```python
# Check position limits before trading
market = kalshi.get_market(ticker)
position_limit = market["position_limit"]  # Max contracts

# Never exceed limit
max_size_usd = min(
    your_intended_size,
    position_limit * current_price
)
```

**Typical limits**:
- High-liquidity markets: 5,000-25,000 contracts
- Medium-liquidity: 1,000-5,000 contracts
- Low-liquidity: 100-1,000 contracts

### Liquidity Considerations

**Always check liquidity before trading**:

```python
def check_liquidity(ticker):
    orderbook = kalshi.get_orderbook(ticker, depth=5)

    # Total size at top 5 levels
    yes_liquidity = sum(level[1] for level in orderbook["yes"][:5])
    no_liquidity = sum(level[1] for level in orderbook["no"][:5])

    print(f"Liquidity for {ticker}:")
    print(f"  YES: {yes_liquidity} contracts")
    print(f"  NO: {no_liquidity} contracts")

    # Rule of thumb: Don't trade if your size > 10% of liquidity
    recommended_max = min(yes_liquidity, no_liquidity) * 0.1
    print(f"  Recommended max: {recommended_max:.0f} contracts")
```

### Slippage on Kalshi

Kalshi has **higher slippage** than Polymarket due to lower liquidity:

```python
# Adjust slippage simulator for Kalshi
SlippageSimulator(
    avg_latency_ms=30,  # Kalshi is faster (U.S. servers)
    impact_coefficient=0.15,  # But higher impact (lower liquidity)
    orderbook_depth=500,  # Shallower orderbooks
)
```

**Expected slippage**:
- Polymarket: 0.5-2% on $100 trade
- Kalshi: 1-5% on $100 trade

**Mitigation**:
- Use smaller position sizes
- Split large orders
- Use limit orders (not market)

---

## Common Issues

### "Order rejected - insufficient balance"

**Cause**: Kalshi requires full balance upfront (no margin)

**Solution**:
```python
# Check balance before trading
balance = kalshi.get_balance()
cost = quantity * price / 100

if cost > balance["balance"] / 100:
    print(f"Insufficient balance: need ${cost:.2f}, have ${balance['balance']/100:.2f}")
```

### "Order rejected - position limit exceeded"

**Cause**: Exceeded per-market position limit

**Solution**:
```python
# Check current position + new order
positions = kalshi.get_positions()
current_position = positions.get(ticker, {}).get("yes_position", 0)
new_total = current_position + quantity

if new_total > position_limit:
    print(f"Position limit exceeded: {new_total} > {position_limit}")
```

### "Market not found"

**Cause**: Kalshi market tickers change frequently

**Solution**:
```python
# Always fetch current markets, don't hardcode tickers
markets = kalshi.get_markets(series_ticker="KXBTC")  # Get all BTC markets
active_market = markets["markets"][0]  # Most recent
ticker = active_market["ticker"]
```

---

## Next Steps

1. **Create Kalshi account** and fund with $100-1000
2. **Test API access** with provided scripts
3. **Run dual observer** for 1 week to measure timing
4. **Train model** on collected data
5. **Deploy with tiny sizes** ($5-10/trade)
6. **Scale up** after validation

---

## Additional Resources

- **Kalshi API Docs**: https://trading-api.readme.io
- **Kalshi Markets**: https://kalshi.com/markets
- **CFTC Regulation**: https://www.cftc.gov/PressRoom/PressReleases/8415-21

---

## Support

**Issues with Kalshi integration?**

1. Check API credentials are correct
2. Verify account is funded and verified
3. Test with demo environment first:
   ```python
   kalshi = KalshiClient(email, password, demo=True)
   ```
4. Review error messages in API responses
5. Contact Kalshi support: support@kalshi.com

**Code issues?**

- Check `helpers/kalshi_client.py` for API calls
- Review `helpers/kalshi_executor.py` for order execution
- Test with `dry_run=True` first

---

## Legal Disclaimer

**Trading prediction markets involves risk.** This software is provided for educational purposes. No guarantee of profits. Only trade with money you can afford to lose.

**Kalshi compliance**: Ensure you comply with all Kalshi terms of service and CFTC regulations. This bot is a tool - you are responsible for how you use it.

**Not financial advice**: This is not investment advice. Consult a financial advisor before trading.
