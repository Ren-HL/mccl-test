#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

read -r -p "请输入要编译的.mu文件名 (例如: odpt_allreduce.mu 或 odpt_allgather.mu): " source_file
if [[ -z "${source_file}" ]]; then
  echo "未提供文件名，已退出。"
  exit 1
fi

if [[ ! -f "${source_file}" ]]; then
  echo "文件不存在: ${source_file}"
  exit 1
fi

# 提取输出程序名（去掉路径和扩展名）
binary_name="$(basename "${source_file}")"
binary_name="${binary_name%.*}"
if [[ -z "${binary_name}" ]]; then
  echo "无法解析输出程序名。"
  exit 1
fi

# 编译时自动链接公共模块 common.mu
srun mcc common.mu "${source_file}" -L"$HOME"/local/lib -Wl,-l:libstdc++.so.6 -L/usr/local/musa/lib -lmccl -lmusart -o "${binary_name}"

# 运行
srun --partition=mt --nodes=1 --gres=gpu:mt:8 --ntasks=1 --cpus-per-task=16 --mem=256G --time=00:20:00 "./${binary_name}"
