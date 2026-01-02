# AWS Observer Setup Guide

**Complete step-by-step guide to deploy t3.micro observers for 24/7 data collection.**

Cost: **$7.30/month per market** (or $1.50/month for shared instance)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Setup (One-time)](#aws-setup-one-time)
3. [Deploy Observer Instance](#deploy-observer-instance)
4. [Verify Data Collection](#verify-data-collection)
5. [Mac Training Setup](#mac-training-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Machine

```bash
# Install AWS CLI
brew install awscli  # Mac
# or
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify
aws --version
```

### AWS Account

1. Sign up at https://aws.amazon.com
2. Create IAM user with programmatic access
3. Attach policies: `AmazonEC2FullAccess`, `AmazonS3FullAccess`
4. Save access key ID and secret

### Configure AWS CLI

```bash
aws configure

# Enter:
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: us-east-1
# Default output format: json
```

---

## AWS Setup (One-time)

### Step 1: Create S3 Bucket for Data

```bash
# Create bucket (must be globally unique name)
aws s3 mb s3://YOUR-USERNAME-market-data --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
    --bucket YOUR-USERNAME-market-data \
    --versioning-configuration Status=Enabled

# Set lifecycle policy to save costs (optional)
cat > /tmp/lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "Archive old data",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket YOUR-USERNAME-market-data \
    --lifecycle-configuration file:///tmp/lifecycle.json
```

**Cost Savings**:
- Standard: $0.023/GB/month
- Standard-IA (after 30 days): $0.0125/GB/month
- Glacier (after 90 days): $0.004/GB/month

### Step 2: Create SSH Key Pair

```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name observer-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/observer-key.pem

# Set permissions
chmod 400 ~/.ssh/observer-key.pem

# Verify
ls -l ~/.ssh/observer-key.pem
```

### Step 3: Create Security Group

```bash
# Create security group
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name observer-sg \
    --description "Security group for market observers" \
    --query 'GroupId' \
    --output text)

echo "Security Group ID: $SECURITY_GROUP_ID"

# Allow SSH from your IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32

# Allow all outbound (default)
echo "✓ Security group configured"
```

### Step 4: Create IAM Role for EC2 (Recommended)

**Instead of storing AWS credentials on instance, use IAM role:**

```bash
# Create trust policy
cat > /tmp/trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
    --role-name ObserverRole \
    --assume-role-policy-document file:///tmp/trust-policy.json

# Attach S3 write policy
cat > /tmp/s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-USERNAME-market-data/*",
        "arn:aws:s3:::YOUR-USERNAME-market-data"
      ]
    }
  ]
}
EOF

aws iam put-role-policy \
    --role-name ObserverRole \
    --policy-name S3Access \
    --policy-document file:///tmp/s3-policy.json

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name ObserverInstanceProfile

aws iam add-role-to-instance-profile \
    --instance-profile-name ObserverInstanceProfile \
    --role-name ObserverRole

echo "✓ IAM role created"
```

---

## Deploy Observer Instance

### Option 1: Automated Deployment (Recommended)

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/cross-market-state-fusion
cd cross-market-state-fusion

# Deploy observer for specific market
export KEY_NAME=observer-key
export SECURITY_GROUP=$SECURITY_GROUP_ID

./scripts/deploy_observer_aws.sh TRUMP2024 YOUR-USERNAME-market-data

# Or deploy for all markets
./scripts/deploy_observer_aws.sh all YOUR-USERNAME-market-data
```

### Option 2: Manual Deployment

#### 2.1: Launch Instance

```bash
# Launch t3.micro
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name observer-key \
    --security-group-ids $SECURITY_GROUP_ID \
    --iam-instance-profile Name=ObserverInstanceProfile \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=observer-trump}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Public IP: $PUBLIC_IP"
```

#### 2.2: Setup Instance

```bash
# SSH into instance
ssh -i ~/.ssh/observer-key.pem ubuntu@$PUBLIC_IP

# Update and install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install Python packages
pip3 install boto3 numpy requests websockets

# Clone repository
git clone https://github.com/YOUR_USERNAME/cross-market-state-fusion
cd cross-market-state-fusion

# Test observer (Ctrl+C to stop)
python3 observer.py --markets TRUMP2024 --s3-bucket YOUR-USERNAME-market-data --tick 0.5
```

#### 2.3: Create Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/observer.service
```

Paste:
```ini
[Unit]
Description=Market Observer
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cross-market-state-fusion
ExecStart=/usr/bin/python3 observer.py --markets TRUMP2024 --s3-bucket YOUR-USERNAME-market-data --tick 0.5
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable observer
sudo systemctl start observer

# Check status
sudo systemctl status observer

# View logs
sudo journalctl -u observer -f
```

---

## Verify Data Collection

### Check Logs

```bash
# SSH to instance
ssh -i ~/.ssh/observer-key.pem ubuntu@$PUBLIC_IP

# View real-time logs
sudo journalctl -u observer -f

# Expected output:
# ✓ Iteration 100 | 3 markets | 250 buffered | 2.0 ticks/sec
# 💾 [12345678] Wrote 1000 observations to obs_20250102_143052.json.gz
# ☁️  [12345678] Uploaded to s3://...
```

### Check S3 Bucket

```bash
# List uploaded files
aws s3 ls s3://YOUR-USERNAME-market-data/market-data/ --recursive

# Expected output:
# market-data/CONDITION_ID_1/2025-01-02/obs_20250102_143052.json.gz
# market-data/CONDITION_ID_2/2025-01-02/obs_20250102_143127.json.gz

# Download a sample file to verify
aws s3 cp s3://YOUR-USERNAME-market-data/market-data/CONDITION_ID_1/2025-01-02/obs_20250102_143052.json.gz /tmp/
gunzip /tmp/obs_20250102_143052.json.gz
cat /tmp/obs_20250102_143052.json | jq '.' | head -50
```

### Monitor Costs

```bash
# Check EC2 costs (after 24 hours)
aws ce get-cost-and-usage \
    --time-period Start=2025-01-01,End=2025-01-03 \
    --granularity DAILY \
    --metrics "UnblendedCost" \
    --group-by Type=DIMENSION,Key=SERVICE

# Expected: ~$0.24/day for t3.micro
```

---

## Mac Training Setup

### Install AWS CLI on Mac

```bash
brew install awscli
aws configure  # Use same credentials as before
```

### Create Training Script

```bash
#!/bin/bash
# sync_and_train.sh - Run hourly via cron

cd ~/trading

# Sync data from S3
echo "Syncing data from S3..."
aws s3 sync s3://YOUR-USERNAME-market-data/market-data/ ./data/

# Train model
echo "Training model..."
python offline_trainer.py \
    --data-dir ./data \
    --output ./models/latest_$(date +%Y%m%d) \
    --epochs 50

# Upload best model
echo "Uploading model..."
aws s3 cp ./models/latest_$(date +%Y%m%d)/ \
    s3://YOUR-USERNAME-market-data/models/latest/ --recursive

echo "✓ Training complete"
```

### Schedule Training (Cron)

```bash
# Open crontab
crontab -e

# Add these lines:
# Sync data every hour
0 * * * * cd ~/trading && aws s3 sync s3://YOUR-USERNAME-market-data/market-data/ ./data/

# Train model every 6 hours
0 */6 * * * cd ~/trading && bash sync_and_train.sh >> ~/trading/logs/training.log 2>&1

# Cleanup old data (keep last 30 days)
0 2 * * * find ~/trading/data -type f -mtime +30 -delete
```

### Monitor Training

```bash
# View training logs
tail -f ~/trading/logs/training.log

# Check latest model
ls -lh ~/trading/models/
```

---

## Troubleshooting

### Observer Not Uploading to S3

**Symptom**: Logs show "S3 upload failed"

**Solution**:
```bash
# Check IAM role is attached
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].IamInstanceProfile'

# If missing, attach IAM role
aws ec2 associate-iam-instance-profile \
    --instance-id $INSTANCE_ID \
    --iam-instance-profile Name=ObserverInstanceProfile

# Restart observer
ssh -i ~/.ssh/observer-key.pem ubuntu@$PUBLIC_IP
sudo systemctl restart observer
```

### Observer Crashing

**Symptom**: Service keeps restarting

**Solution**:
```bash
# Check logs for errors
sudo journalctl -u observer -n 100

# Common issues:
# 1. Network error - check security group allows outbound
# 2. Missing dependencies - reinstall: pip3 install boto3 numpy requests websockets
# 3. Invalid market - check markets exist: python3 observer.py --markets all --help
```

### No Data Being Collected

**Symptom**: Buffered observations always 0

**Solution**:
```bash
# Test manually
python3 observer.py --markets all --local-dir /tmp/test --tick 0.5

# Check if markets are available
# If no markets found, check Polymarket API is accessible
curl -s https://clob.polymarket.com/markets | jq '.' | head
```

### High Costs

**Symptom**: Monthly bill > $10/instance

**Solution**:
```bash
# Check if instance is right type
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].InstanceType'

# Should be: t3.micro
# If larger, terminate and redeploy:
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
./scripts/deploy_observer_aws.sh TRUMP2024 YOUR-USERNAME-market-data
```

### SSH Connection Timeout

**Symptom**: Can't SSH to instance

**Solution**:
```bash
# Check instance is running
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].State.Name'

# Check security group allows your IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "Your IP: $MY_IP"

# Update security group
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32
```

---

## Scaling to Multiple Markets

### Deploy Multiple Observers (Separate Instances)

```bash
# Deploy one instance per market
./scripts/deploy_observer_aws.sh TRUMP2024 YOUR-USERNAME-market-data
./scripts/deploy_observer_aws.sh ETH10K YOUR-USERNAME-market-data
./scripts/deploy_observer_aws.sh BTC100K YOUR-USERNAME-market-data

# Cost: $7.30 × 3 = $21.90/month
```

### Deploy Single Instance for Multiple Markets

```bash
# Launch t3.small (2 vCPU, 2GB RAM) - $15/month
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.small \
    --key-name observer-key \
    --security-group-ids $SECURITY_GROUP_ID \
    --iam-instance-profile Name=ObserverInstanceProfile \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=observer-multi}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

# Setup and run multiple observers
ssh -i ~/.ssh/observer-key.pem ubuntu@$PUBLIC_IP

# Run each market as separate process
nohup python3 observer.py --markets TRUMP2024 --s3-bucket YOUR-USERNAME-market-data > trump.log 2>&1 &
nohup python3 observer.py --markets ETH10K --s3-bucket YOUR-USERNAME-market-data > eth.log 2>&1 &
nohup python3 observer.py --markets BTC100K --s3-bucket YOUR-USERNAME-market-data > btc.log 2>&1 &

# Or observe all markets in single process
nohup python3 observer.py --markets all --s3-bucket YOUR-USERNAME-market-data > all.log 2>&1 &

# Cost: $15/month for 10-20 markets
```

---

## Cost Summary

### Per-Market (Separate Instances)

```
t3.micro (1 vCPU, 1GB):     $7.30/month
S3 storage (10GB/month):    $0.23/month
S3 PUT requests (100K):     $0.50/month
Data transfer (negligible): $0.10/month
─────────────────────────────────────
Total per market:           $8.13/month
```

### Multi-Market (Shared Instance)

```
t3.small (2 vCPU, 2GB):     $15.04/month
S3 storage (50GB/month):    $1.15/month
S3 PUT requests (500K):     $2.50/month
Data transfer (negligible): $0.50/month
─────────────────────────────────────
Total for 5-10 markets:     $19.19/month
Per market:                 $2-4/month
```

### Mac Mini Training

```
Electricity (20W idle):     $1-2/month
Total cost:                 FREE (you already have it)
```

---

## Next Steps

1. **Week 1-2**: Collect data
   - Deploy observers
   - Verify S3 uploads
   - Let run 24/7

2. **Week 2+**: Start training
   - Sync data to Mac
   - Run offline training
   - Evaluate models

3. **Month 1+**: Deploy live
   - Convert model to PyTorch
   - Deploy to AWS EC2 for live trading
   - Start with tiny sizes ($5-10)

---

## Support

**Issues?**
- Check logs: `sudo journalctl -u observer -f`
- Test manually: `python3 observer.py --markets all --local-dir /tmp/test`
- Verify AWS: `aws s3 ls` and `aws ec2 describe-instances`

**Questions?**
- Open issue on GitHub
- Check AWS documentation: https://docs.aws.amazon.com/ec2/
