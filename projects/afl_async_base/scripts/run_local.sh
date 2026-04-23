#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"

usage() {
  cat <<'EOF'
用法: run_local.sh [CONFIG]

  CONFIG 可选：
    smoke | base     使用项目内 configs/<名字>.yaml
    其它路径         任意 .yaml；可先写相对 ROOT 的路径，再尝试当前工作目录

  默认：base（与未设置 AFL_CONFIG 时 server 的默认配置一致）

示例：
  ./scripts/run_local.sh smoke
  ./scripts/run_local.sh base
  ./scripts/run_local.sh configs/smoke.yaml
  AFL_CONFIG=/path/to/custom.yaml ./scripts/run_local.sh   # 仍尊重已导出的 AFL_CONFIG
EOF
  exit "${1:-0}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

# 已手动导出 AFL_CONFIG 时，不再解析第一个参数（便于与其它脚本链接）
if [[ -n "${AFL_CONFIG:-}" ]]; then
  if [[ ! -f "$AFL_CONFIG" ]]; then
    echo "AFL_CONFIG 指向的文件不存在: $AFL_CONFIG" >&2
    exit 1
  fi
  echo "[run_local] AFL_CONFIG=$AFL_CONFIG (from environment)"
  exec python -m afl.server
fi

NAME="${1:-base}"
CONFIG_PATH=""

if [[ -f "$NAME" ]]; then
  CONFIG_PATH="$(realpath "$NAME")"
elif [[ -f "$ROOT/$NAME" ]]; then
  CONFIG_PATH="$(realpath "$ROOT/$NAME")"
elif [[ -f "$ROOT/configs/${NAME}.yaml" ]]; then
  CONFIG_PATH="$ROOT/configs/${NAME}.yaml"
else
  echo "找不到配置: $NAME" >&2
  echo "试过: 当前目录、$ROOT/$NAME、$ROOT/configs/${NAME}.yaml" >&2
  usage 1
fi

export AFL_CONFIG="$CONFIG_PATH"
echo "[run_local] AFL_CONFIG=$AFL_CONFIG"
python -m afl.server
