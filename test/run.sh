#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

read -r -p "请输入要运行的程序文件名 (例如: allreduce_perf 或 allgather_perf): " target
if [[ -z "${target}" ]]; then
  echo "未提供文件名，已退出。"
  exit 1
fi

if [[ ! -f "${target}" ]]; then
  echo "文件不存在: ${target}"
  exit 1
fi

read -r -p "请输入运行参数(可留空): " run_args

srun --partition=mt --nodes=1 --gres=gpu:mt:8 --ntasks=1 --cpus-per-task=16 \
  --mem=256G --time=00:20:00 "./${target}" ${run_args}
