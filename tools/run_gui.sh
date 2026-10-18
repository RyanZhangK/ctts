#!/usr/bin/env sh
# 启动 CTTS GUI；环境缺失时自动创建并安装依赖
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/setup_env.py" --run "$@"
