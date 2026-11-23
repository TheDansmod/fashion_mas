#!/bin/bash
set -e
exec > /var/log/userdata.log 2>&1

# 1. System update and base tools
dnf update -y
dnf install -y git tar gzip

# 2. Install Podman (available in AL2023 via SPAL repo)
dnf install -y spal-release
dnf install -y podman podman-compose

# 3. Install uv (standalone installer, works on ARM64)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'source $HOME/.local/bin/env' >> /root/.bashrc
source /root/.local/bin/env

# 4. Clone your public repo
mkdir -p /opt/app
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git /opt/app
cd /opt/app

# 5. Install Python dependencies via uv
uv sync

echo "Bootstrap complete."
