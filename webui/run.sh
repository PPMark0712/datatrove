#!/bin/bash
set -euo pipefail
# ============================================================
# Datatrove WebUI 启动脚本
# ============================================================

# 账号密码设置（格式: "用户名:密码"，多个用逗号分隔）
# 例: "admin:mypass123,guest:guest456"
export WEBUI_AUTH="admin:datatrove2024"

# 服务端口
export WEBUI_PORT=7860

# GPU 设置（CDF-GC 依存分析需要 GPU，按需修改）
# export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$(dirname "$0")"

if ! command -v python &> /dev/null; then
    echo "Error: python not found. Please activate your conda environment first."
    exit 1
fi

echo "Starting Datatrove WebUI on port ${WEBUI_PORT}..."
echo "Login: $(echo "${WEBUI_AUTH}" | cut -d: -f1):****"
python app.py
