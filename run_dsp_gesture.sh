#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f SHA256SUMS.txt ]; then
    printf 'ERROR: SHA256SUMS.txt is missing. Extract the complete release again.\n' >&2
    exit 1
fi
if command -v sha256sum >/dev/null 2>&1 && ! sha256sum -c SHA256SUMS.txt >/dev/null; then
    printf 'ERROR: File verification failed. Extract the complete release again.\n' >&2
    exit 1
fi

for file in \
    dsp_live_gesture/live.py \
    dsp_live_gesture/burst_engine.py \
    dsp_live_gesture/engine_base.py \
    dsp_live_gesture/view_rd.py \
    dsp_live_gesture/ui_compact.html \
    dsp_live_gesture/ui_full.html \
    preprocessing_context17.py \
    preprocessing_efcir.py \
    checkpoints/yuqing20_standalone_seed42.pt \
    checkpoints/efcir_seed42.pt
do
    if [ ! -s "$file" ]; then
        printf 'ERROR: Missing or empty file: %s\n' "$file" >&2
        exit 1
    fi
done

task_python="$HOME/venvs/lptf-nano-torch110/bin/python3"
if [ ! -x "$task_python" ]; then
    printf 'ERROR: PyTorch 1.10 environment is missing: %s\n' "$task_python" >&2
    exit 1
fi
torch_version=$("$task_python" -c 'import torch; print(torch.__version__)')
case "$torch_version" in
    1.10*) ;;
    *) printf 'ERROR: Expected PyTorch 1.10, found %s\n' "$torch_version" >&2; exit 1 ;;
esac

printf 'Python: %s\nPyTorch: %s\n' "$task_python" "$torch_version"
printf 'Defaults: --model context17 --ui compact\n'
export OPENBLAS_CORETYPE=ARMV8
export NANO_EVAL_BATCH=1
export PYTHONIOENCODING=utf-8
exec "$task_python" -u dsp_live_gesture/live.py "$@"
