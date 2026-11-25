#!/usr/bin/env bash
# Run by the user on the Linux SERVER. Default mode only prints information.
set -euo pipefail

mode="${1:---probe}"
qat_env="${2:-gesture_aimet_qat}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$mode" != "--probe" && "$mode" != "--install" && "$mode" != "--check" ]]; then
  echo "Usage: bash qat_plugin/server_setup_2026-11-24.sh [--probe|--install|--check] [gesture_aimet_qat]" >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for the Linux training server. No download or installation was started." >&2
  exit 2
fi
cd "$project_root"

if [[ "$mode" == "--probe" ]]; then
  echo "Read-only server probe; no packages will be downloaded."
  python --version
  python -c 'import sys, platform; print("Python:", sys.executable); print("OS:", platform.platform())'
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
  python qat_plugin/run_2026-11-24.py --preflight || true
  exit 0
fi

if [[ ! -f "2026-11-24_QAT_install_manifest.json" ]]; then
  echo "First create gesture_train_shared_QAT using install_2026-11-24.py, then run this script there." >&2
  exit 2
fi
if [[ ! "$qat_env" =~ ^gesture_aimet_qat(_[A-Za-z0-9_-]+)?$ ]]; then
  echo "Use a dedicated gesture_aimet_qat environment; base/yolov5 are never modified." >&2
  exit 2
fi

conda_tool="${CONDA_EXE:-}"
if [[ -z "$conda_tool" ]]; then conda_tool="$(type -P conda || true)"; fi
if [[ -z "$conda_tool" && -x /home/sunyu/anaconda3/bin/conda ]]; then
  conda_tool=/home/sunyu/anaconda3/bin/conda
fi
if [[ ! -x "$conda_tool" ]]; then
  echo "Conda not found. Set CONDA_EXE to its absolute executable path; nothing installed." >&2
  exit 2
fi

resolve_env() {
  "$conda_tool" env list --json | python -c \
    'import json,os,sys; matches=[p for p in json.load(sys.stdin)["envs"] if os.path.basename(p)==sys.argv[1]]; assert len(matches)<=1, "Ambiguous Conda environment name"; print(matches[0] if matches else "")' "$qat_env"
}

env_prefix="$(resolve_env)"
if [[ -z "$env_prefix" ]]; then
  if [[ "$mode" != "--install" ]]; then
    echo "Environment not found: $qat_env. Run --install on the server first." >&2
    exit 2
  fi
  "$conda_tool" create -y -n "$qat_env" python=3.10 pip
  env_prefix="$(resolve_env)"
fi
if [[ -z "$env_prefix" || ! -x "$env_prefix/bin/python" ]]; then
  echo "Cannot locate isolated environment Python: $qat_env. No pip command executed." >&2
  exit 2
fi
qat_python="$env_prefix/bin/python"
echo "Using isolated environment Python: $qat_python"
# Older Conda versions do not support --no-capture-output. Avoid conda run entirely.
"$qat_python" -c 'import sys; assert sys.version_info >= (3,10), "Existing environment uses Python <3.10; choose a new dedicated name"'

if [[ "$mode" == "--install" ]]; then
  echo "Dependencies will be downloaded on THIS SERVER into isolated environment: $qat_env"
  # Explicit wheel choice for the older server GPUs; the smoke test must still confirm driver compatibility.
  "$qat_python" -m pip install \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
  "$qat_python" -m pip install \
    -r qat_plugin/requirements_2026-11-24.txt
  "$qat_python" -m pip check
  "$qat_python" qat_plugin/run_2026-11-24.py --preflight
  echo "Installed. Next run --check, then the real-data --smoke command from the README."
else
  # CPU tests do not occupy either GPU or train on the real dataset.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$qat_python" -m unittest discover -s qat_plugin/tests -v
fi
