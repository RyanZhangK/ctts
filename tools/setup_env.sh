#!/usr/bin/env sh
# CTTS 跨平台环境初始化（Linux / macOS 入口）
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/setup_env.py" "$@"
