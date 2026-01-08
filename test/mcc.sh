#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

read -r -p "请输入要编译的.mu文件名 (例如: allreduce.mu 或 allgather.mu): " source_file
if [[ -z "${source_file}" ]]; then
  echo "未提供文件名，已退出。"
  exit 1
fi

if [[ ! -f "${source_file}" ]]; then
  echo "文件不存在: ${source_file}"
  exit 1
fi

binary_name="$(basename "${source_file}")"
binary_name="${binary_name%.*}_perf"
if [[ -z "${binary_name}" ]]; then
  echo "无法解析输出程序名。"
  exit 1
fi

srun mcc common.mu "${source_file}" -L"$HOME"/local/lib -Wl,-l:libstdc++.so.6 \
  -L/usr/local/musa/lib -lmccl -lmusart -o "${binary_name}"

echo "已生成: ${binary_name}"
