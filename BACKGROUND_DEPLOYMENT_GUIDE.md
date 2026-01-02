# Background & Cloud Deployment Guide

**Goal**: Run the trading bot 24/7 in the background (locally or cloud) with automatic restarts, monitoring, and alerting.

---

## Option 1: Local Background Execution (Mac/Linux)

### Using systemd (Linux) or launchd (Mac)

**Best for**: Running on your local machine 24/7

#### Mac Setup (launchd)

**Step 1: Create Launch Agent**

```bash
# Create plist file
cat > ~/Library/LaunchAgents/com.polymarket.trading.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.polymarket.trading</string>

    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/python</string>
        <string>/path/to/cross-market-state-fusion/run.py</string>
        <string>rl</string>
        <string>--load</string>
        <string>rl_model</string>
        <string>--size</string>
        <string>50</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>POLYMARKET_KEY</key>
        <string>0x1234...</string>  <!-- Your private key -->
    </dict>

    <key>WorkingDirectory</key>
    <string>/path/to/cross-market-state-fusion</string>

    <key>StandardOutPath</key>
    <string>/tmp/polymarket-trading.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/polymarket-trading-error.log</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>  <!-- Restart on crash -->
    </dict>

    <key>ThrottleInterval</key>
    <integer>60</integer>  <!-- Wait 60s before restarting -->
</dict>
</plist>
EOF

# Load the agent
launchctl load ~/Library/LaunchAgents/com.polymarket.trading.plist

# Start it
launchctl start com.polymarket.trading

# Check status
launchctl list | grep polymarket

# View logs
tail -f /tmp/polymarket-trading.log
```

**Control commands**:
```bash
# Stop
launchctl stop com.polymarket.trading

# Restart
launchctl stop com.polymarket.trading && launchctl start com.polymarket.trading

# Unload (disable autostart)
launchctl unload ~/Library/LaunchAgents/com.polymarket.trading.plist
```

#### Linux Setup (systemd)

**Step 1: Create systemd service**

```bash
# Create service file
sudo cat > /etc/systemd/system/polymarket-trading.service <<'EOF'
[Unit]
Description=Polymarket Trading Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/cross-market-state-fusion
Environment="POLYMARKET_KEY=0x1234..."
ExecStart=/home/your-username/cross-market-state-fusion/venv/bin/python run.py rl --load rl_model --size 50
Restart=always
RestartSec=60
StandardOutput=append:/var/log/polymarket-trading.log
StandardError=append:/var/log/polymarket-trading-error.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable (start on boot)
sudo systemctl enable polymarket-trading

# Start now
sudo systemctl start polymarket-trading

# Check status
sudo systemctl status polymarket-trading

# View logs
sudo journalctl -u polymarket-trading -f
```

**Control commands**:
```bash
# Stop
sudo systemctl stop polymarket-trading

# Restart
sudo systemctl restart polymarket-trading

# Disable (no autostart)
sudo systemctl disable polymarket-trading
```

---

## Option 2: Docker Container (Portable)

**Best for**: Easy deployment anywhere (local or cloud)

### Dockerfile

```dockerfile
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlx>=0.5.0 \
    websockets>=12.0 \
    flask>=3.0.0 \
    flask-socketio>=5.3.0 \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    eth-account \
    web3

# Run the bot
CMD ["python", "run.py", "rl", "--load", "rl_model", "--size", "50"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  trading-bot:
    build: .
    container_name: polymarket-trading
    restart: unless-stopped
    environment:
      - POLYMARKET_KEY=${POLYMARKET_KEY}
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Usage

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Restart
docker-compose restart

# Run with live trading
POLYMARKET_KEY="0x123..." docker-compose up -d
```

---

## Option 3: Cloud Deployment (AWS EC2)

**Best for**: Low latency (near Polymarket infrastructure)

### Quick Deploy Script

Create `deploy_aws.sh`:

```bash
#!/bin/bash

# AWS EC2 deployment script
# Run: ./deploy_aws.sh

echo "🚀 Deploying to AWS EC2..."

# Configuration
INSTANCE_TYPE="t3.medium"
REGION="us-east-1"
AMI="ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 LTS
KEY_NAME="your-ssh-key"

# Launch instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --region $REGION \
  --security-groups "polymarket-trading-sg" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance IP: $PUBLIC_IP"

# Wait for SSH to be ready
echo "Waiting for SSH..."
sleep 30

# Copy code to instance
echo "Copying code..."
scp -r ../cross-market-state-fusion ubuntu@$PUBLIC_IP:~/

# SSH and setup
echo "Setting up instance..."
ssh ubuntu@$PUBLIC_IP << 'EOF'
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Setup app
cd ~/cross-market-state-fusion
python3 -m venv venv
source venv/bin/activate
pip install mlx websockets flask flask-socketio numpy requests eth-account web3

# Create systemd service
sudo tee /etc/systemd/system/polymarket-trading.service > /dev/null <<'SERVICE'
[Unit]
Description=Polymarket Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cross-market-state-fusion
Environment="POLYMARKET_KEY=YOUR_KEY_HERE"
ExecStart=/home/ubuntu/cross-market-state-fusion/venv/bin/python run.py rl --load rl_model --size 50
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
SERVICE

# Start service
sudo systemctl daemon-reload
sudo systemctl enable polymarket-trading
sudo systemctl start polymarket-trading
EOF

echo "✅ Deployment complete!"
echo "Instance IP: $PUBLIC_IP"
echo "SSH: ssh ubuntu@$PUBLIC_IP"
echo "Check status: ssh ubuntu@$PUBLIC_IP 'sudo systemctl status polymarket-trading'"
```

### Manual AWS Setup

**Step 1: Launch EC2 Instance**

```bash
# Via AWS Console:
# 1. Go to EC2 → Launch Instance
# 2. Choose Ubuntu 22.04 LTS
# 3. Instance type: t3.medium (2 vCPU, 4GB RAM)
# 4. Region: us-east-1 (Virginia)
# 5. Create/select key pair
# 6. Security group: Allow SSH (22) from your IP

# Or via CLI:
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key \
  --region us-east-1
```

**Step 2: Connect and Setup**

```bash
# SSH to instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Clone repo
git clone https://github.com/your-username/cross-market-state-fusion.git
cd cross-market-state-fusion

# Setup venv
python3 -m venv venv
source venv/bin/activate
pip install mlx websockets flask flask-socketio numpy requests eth-account web3

# Set private key (secure method)
export POLYMARKET_KEY="0x1234..."

# Test run
python run.py rl --load rl_model --size 50
```

**Step 3: Create systemd service** (see Linux setup above)

**Step 4: Configure monitoring** (see Monitoring section below)

---

## Process Management with Supervisor (Alternative)

**Best for**: More control than systemd, works on any Linux

### Install Supervisor

```bash
sudo apt-get install supervisor
```

### Create config

```bash
sudo cat > /etc/supervisor/conf.d/polymarket-trading.conf <<'EOF'
[program:polymarket-trading]
command=/home/ubuntu/cross-market-state-fusion/venv/bin/python run.py rl --load rl_model --size 50
directory=/home/ubuntu/cross-market-state-fusion
user=ubuntu
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/polymarket-trading.err.log
stdout_logfile=/var/log/polymarket-trading.out.log
environment=POLYMARKET_KEY="0x1234..."
EOF

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Start
sudo supervisorctl start polymarket-trading

# Status
sudo supervisorctl status polymarket-trading
```

**Control commands**:
```bash
sudo supervisorctl stop polymarket-trading
sudo supervisorctl restart polymarket-trading
sudo supervisorctl tail -f polymarket-trading
```

---

## Monitoring & Alerting

### 1. Health Check Script

Create `healthcheck.py`:

```python
#!/usr/bin/env python3
"""
Health check script for monitoring trading bot.
Run via cron every 5 minutes.
"""
import os
import sys
import requests
import subprocess
from datetime import datetime, timedelta

# Configuration
LOG_FILE = "/var/log/polymarket-trading.log"
MAX_LOG_AGE_MINUTES = 10  # Alert if no log activity in 10 min
ALERT_EMAIL = "your-email@gmail.com"

def check_process_running():
    """Check if trading process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run.py"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except Exception as e:
        return False

def check_log_freshness():
    """Check if log file has recent activity."""
    try:
        if not os.path.exists(LOG_FILE):
            return False

        mtime = os.path.getmtime(LOG_FILE)
        age = datetime.now().timestamp() - mtime
        return age < (MAX_LOG_AGE_MINUTES * 60)
    except Exception as e:
        return False

def send_alert(message):
    """Send alert (email or SMS)."""
    print(f"🚨 ALERT: {message}")

    # Option 1: Email via SendGrid/SES
    # requests.post(...)

    # Option 2: SMS via Twilio
    # requests.post(...)

    # Option 3: Slack webhook
    # webhook_url = "https://hooks.slack.com/services/..."
    # requests.post(webhook_url, json={"text": message})

def main():
    alerts = []

    # Check if process is running
    if not check_process_running():
        alerts.append("Trading bot process not running")

    # Check log freshness
    if not check_log_freshness():
        alerts.append(f"No log activity in {MAX_LOG_AGE_MINUTES} minutes")

    # Send alerts
    if alerts:
        for alert in alerts:
            send_alert(alert)
        sys.exit(1)
    else:
        print("✅ All checks passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

### 2. Cron Job for Health Checks

```bash
# Edit crontab
crontab -e

# Add health check every 5 minutes
*/5 * * * * /path/to/venv/bin/python /path/to/healthcheck.py >> /var/log/healthcheck.log 2>&1
```

### 3. CloudWatch Monitoring (AWS)

**Install CloudWatch agent**:

```bash
# On EC2 instance
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure
sudo cat > /opt/aws/amazon-cloudwatch-agent/etc/config.json <<'EOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/polymarket-trading.log",
            "log_group_name": "/polymarket/trading",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  },
  "metrics": {
    "metrics_collected": {
      "cpu": {
        "measurement": [{"name": "cpu_usage_idle"}],
        "totalcpu": false
      },
      "mem": {
        "measurement": [{"name": "mem_used_percent"}]
      }
    }
  }
}
EOF

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json
```

**Create CloudWatch alarms**:

```bash
# Alert if no log activity for 10 minutes
aws cloudwatch put-metric-alarm \
  --alarm-name polymarket-trading-inactive \
  --alarm-description "Trading bot inactive" \
  --metric-name IncomingLogEvents \
  --namespace AWS/Logs \
  --statistic Sum \
  --period 600 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 1 \
  --dimensions Name=LogGroupName,Value=/polymarket/trading
```

### 4. Custom Metrics Script

Create `log_metrics.py`:

```python
#!/usr/bin/env python3
"""
Parse trading bot logs and send metrics to CloudWatch.
Run via cron every minute.
"""
import re
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')

def parse_latest_stats():
    """Parse PnL and trade count from logs."""
    with open('/var/log/polymarket-trading.log', 'r') as f:
        lines = f.readlines()[-100:]  # Last 100 lines

    pnl = 0.0
    trades = 0

    for line in reversed(lines):
        # Look for: "PnL: $+12.50 | Trades: 45"
        match = re.search(r'PnL: \$([+-]?\d+\.\d+) \| Trades: (\d+)', line)
        if match:
            pnl = float(match.group(1))
            trades = int(match.group(2))
            break

    return pnl, trades

def send_metrics(pnl, trades):
    """Send metrics to CloudWatch."""
    cloudwatch.put_metric_data(
        Namespace='Polymarket/Trading',
        MetricData=[
            {
                'MetricName': 'PnL',
                'Value': pnl,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'TradeCount',
                'Value': trades,
                'Unit': 'Count',
                'Timestamp': datetime.utcnow()
            }
        ]
    )

if __name__ == "__main__":
    pnl, trades = parse_latest_stats()
    send_metrics(pnl, trades)
    print(f"Sent metrics: PnL=${pnl}, Trades={trades}")
```

**Add to cron**:
```bash
* * * * * /path/to/venv/bin/python /path/to/log_metrics.py
```

---

## Graceful Shutdown & Restart

### Add signal handling to run.py

```python
# Add to run.py
import signal
import asyncio

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n⚠️  Shutdown signal received, closing positions...")
        self.kill_now = True

# In main()
killer = GracefulKiller()

# In decision loop
async def decision_loop(self):
    while self.running and not killer.kill_now:
        # ... existing code ...

    # Cleanup on shutdown
    if killer.kill_now:
        print("Closing all positions before shutdown...")
        self.close_all_positions()

        if isinstance(self.strategy, RLStrategy):
            self.strategy.save("rl_model_latest")
            print("Model saved")
```

### Safe restart script

Create `safe_restart.sh`:

```bash
#!/bin/bash

# Safe restart: close positions, save state, restart

echo "Sending shutdown signal..."
sudo systemctl kill -s TERM polymarket-trading

echo "Waiting for graceful shutdown..."
sleep 10

echo "Restarting..."
sudo systemctl start polymarket-trading

echo "✅ Restart complete"
sudo systemctl status polymarket-trading
```

---

## Backup & Recovery

### Automated backups

```bash
# Backup script (run daily via cron)
#!/bin/bash

BACKUP_DIR="/backups/polymarket"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# Backup model
cp rl_model.safetensors $BACKUP_DIR/rl_model_$DATE.safetensors

# Backup logs
cp /var/log/polymarket-trading.log $BACKUP_DIR/logs_$DATE.log

# Backup trade history (if logging to CSV)
cp trades.csv $BACKUP_DIR/trades_$DATE.csv

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-bucket/polymarket-backups/

# Keep only last 30 days
find $BACKUP_DIR -name "*.safetensors" -mtime +30 -delete

echo "✅ Backup complete: $DATE"
```

**Add to cron**:
```bash
0 2 * * * /path/to/backup.sh >> /var/log/backup.log 2>&1
```

---

## Cost Analysis

### Local Background Execution

```
Hardware: Your Mac (already owned)     $0/month
Electricity: ~5W @ $0.12/kWh           $0.43/month
Internet: Included                      $0/month
----------------------------------------------------
Total:                                 ~$0.50/month
```

**Pros**: Cheapest, no setup
**Cons**: Mac must stay on 24/7, home internet reliability

### Cloud (AWS EC2)

```
EC2 t3.medium (2 vCPU, 4GB):           $30/month
EBS Storage 20GB:                       $2/month
Data Transfer:                          $1/month
CloudWatch:                             $3/month
----------------------------------------------------
Total:                                 ~$36/month
```

**Pros**: Always on, reliable, low latency
**Cons**: Monthly cost, requires setup

### Cloud (AWS EC2 Spot Instance)

```
EC2 t3.medium spot (60-90% discount):  $9/month
EBS Storage 20GB:                       $2/month
Data Transfer:                          $1/month
CloudWatch:                             $3/month
----------------------------------------------------
Total:                                 ~$15/month
```

**Pros**: Cheapest cloud option
**Cons**: Can be interrupted (rare for t3.medium)

---

## Recommended Setup

### For Testing ($0/month)

```bash
# Local Mac with launchd
launchctl load ~/Library/LaunchAgents/com.polymarket.trading.plist
```

**Perfect for**:
- Paper trading
- Tiny size live trading ($5-10)
- Learning and iteration

### For Production ($15-36/month)

```bash
# AWS EC2 Spot Instance + systemd + CloudWatch
```

**Perfect for**:
- 24/7 unattended operation
- Low latency (near Polymarket infrastructure)
- Professional monitoring and alerting
- Scaling to larger sizes ($100-500)

---

## Quick Start Commands

### Local Background (Mac)

```bash
# 1. Setup
cp launchd.plist ~/Library/LaunchAgents/com.polymarket.trading.plist
# Edit plist with your paths and settings

# 2. Start
launchctl load ~/Library/LaunchAgents/com.polymarket.trading.plist

# 3. Monitor
tail -f /tmp/polymarket-trading.log

# 4. Stop
launchctl unload ~/Library/LaunchAgents/com.polymarket.trading.plist
```

### Cloud (AWS EC2)

```bash
# 1. Launch instance
./deploy_aws.sh

# 2. SSH and check status
ssh ubuntu@<instance-ip>
sudo systemctl status polymarket-trading

# 3. View logs
sudo journalctl -u polymarket-trading -f

# 4. Update model (from local Mac)
scp rl_model.safetensors ubuntu@<instance-ip>:~/cross-market-state-fusion/
ssh ubuntu@<instance-ip> 'sudo systemctl restart polymarket-trading'
```

---

## Troubleshooting

### Bot stops running

**Check**:
```bash
# Process status
sudo systemctl status polymarket-trading

# Recent logs
sudo journalctl -u polymarket-trading -n 100

# System resources
top
df -h
```

**Common fixes**:
```bash
# Restart
sudo systemctl restart polymarket-trading

# Check for errors
sudo journalctl -u polymarket-trading --since "1 hour ago" | grep ERROR

# Increase restart throttle if crashing repeatedly
# Edit: RestartSec=300 in service file
```

### High memory usage

```bash
# Check memory
free -h

# Restart if needed
sudo systemctl restart polymarket-trading
```

### Lost connection to WebSocket

```bash
# Check network
ping 8.8.8.8

# Check logs for reconnection attempts
grep "reconnect" /var/log/polymarket-trading.log

# Should auto-reconnect, but restart if stuck
sudo systemctl restart polymarket-trading
```

---

## Security Checklist

- [ ] Private key stored in environment variable (not hardcoded)
- [ ] Log files don't contain sensitive data
- [ ] SSH key-based authentication (no password)
- [ ] Firewall configured (only allow SSH from your IP)
- [ ] Regular backups of model and logs
- [ ] CloudWatch alarms configured
- [ ] Health check cron job running
- [ ] systemd service runs as non-root user

---

## Next Steps

1. **Start local**: Test with launchd/systemd locally
2. **Add monitoring**: Set up health checks and alerts
3. **Deploy to cloud**: Move to EC2 for better latency
4. **Automate backups**: Daily model and log backups
5. **Scale gradually**: Increase size only after proven reliability

**Remember**: Start with paper trading or tiny sizes ($5-10) before scaling up!
