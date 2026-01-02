#!/usr/bin/env bash
#
# Deploy observer to AWS EC2 t3.micro instance
#
# Usage:
#   ./scripts/deploy_observer_aws.sh TRUMP2024 my-data-bucket
#   ./scripts/deploy_observer_aws.sh "all" my-data-bucket
#

set -e

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <market> <s3-bucket> [region]"
    echo ""
    echo "Examples:"
    echo "  $0 TRUMP2024 my-data-bucket"
    echo "  $0 ETH10K my-data-bucket us-east-1"
    echo "  $0 all my-data-bucket"
    exit 1
fi

MARKET="$1"
S3_BUCKET="$2"
REGION="${3:-us-east-1}"

# Configuration
INSTANCE_TYPE="t3.micro"
AMI_ID="ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 LTS (us-east-1)
KEY_NAME="${KEY_NAME:-my-key}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-default}"

echo "=========================================="
echo "Deploying Observer to AWS EC2"
echo "=========================================="
echo "Market: $MARKET"
echo "S3 Bucket: $S3_BUCKET"
echo "Region: $REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo ""

# Generate a unique name for this observer
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_NAME="observer-${MARKET}-${TIMESTAMP}"

echo "Step 1: Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Type,Value=Observer},{Key=Market,Value=${MARKET}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✓ Instance launched: $INSTANCE_ID"
echo "  Waiting for instance to be running..."

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✓ Instance running at: $PUBLIC_IP"
echo "  Waiting for SSH to be ready (30s)..."
sleep 30

echo ""
echo "Step 2: Setting up instance..."

# Create setup script
cat > /tmp/setup_observer.sh << 'EOF'
#!/bin/bash
set -e

echo "Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git awscli

echo "Installing Python packages..."
pip3 install -q boto3 numpy requests websockets

echo "Cloning repository..."
cd /home/ubuntu
git clone https://github.com/YOUR_USERNAME/cross-market-state-fusion.git || true
cd cross-market-state-fusion

echo "✓ Setup complete"
EOF

# Upload and run setup script
scp -o StrictHostKeyChecking=no -i ~/.ssh/${KEY_NAME}.pem /tmp/setup_observer.sh ubuntu@${PUBLIC_IP}:/tmp/
ssh -o StrictHostKeyChecking=no -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'bash /tmp/setup_observer.sh'

echo "✓ Setup complete"

echo ""
echo "Step 3: Configuring AWS credentials..."

# Note: In production, use IAM roles instead
echo "⚠️  You need to configure AWS credentials on the instance"
echo "   Option 1: Attach IAM role with S3 write permissions (recommended)"
echo "   Option 2: SSH in and run 'aws configure'"

echo ""
echo "Step 4: Starting observer..."

# Create systemd service file
cat > /tmp/observer.service << EOF
[Unit]
Description=Market Observer - ${MARKET}
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cross-market-state-fusion
ExecStart=/usr/bin/python3 observer.py --markets ${MARKET} --s3-bucket ${S3_BUCKET} --tick 0.5
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Upload service file
scp -i ~/.ssh/${KEY_NAME}.pem /tmp/observer.service ubuntu@${PUBLIC_IP}:/tmp/
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} << 'ENDSSH'
sudo mv /tmp/observer.service /etc/systemd/system/observer.service
sudo systemctl daemon-reload
sudo systemctl enable observer
sudo systemctl start observer
ENDSSH

echo "✓ Observer started as systemd service"

echo ""
echo "=========================================="
echo "✓ Deployment Complete"
echo "=========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Name: $INSTANCE_NAME"
echo ""
echo "Next steps:"
echo "1. Configure AWS credentials:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "   aws configure  # or attach IAM role"
echo ""
echo "2. Check observer status:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "   sudo systemctl status observer"
echo "   sudo journalctl -u observer -f"
echo ""
echo "3. Monitor S3 uploads:"
echo "   aws s3 ls s3://${S3_BUCKET}/market-data/ --recursive"
echo ""
echo "Monthly cost: ~\$7.30/month (t3.micro)"
