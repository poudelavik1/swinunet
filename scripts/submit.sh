#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/submit.sh /absolute/dataset/path [output-directory]" >&2
    exit 2
fi
export DATASET="$(realpath "$1")"
for split in train val test; do
    for kind in images masks; do
        test -d "$DATASET/$split/$kind" || { echo "Missing $DATASET/$split/$kind" >&2; exit 1; }
    done
done
cd "$ROOT"
mkdir -p logs runs
export OUTPUT_DIR="$(realpath -m "${2:-$ROOT/runs/$(date +%Y%m%d_%H%M%S)_$$}")"
if [[ "${RESUME:-0}" == 1 ]]; then
    test -f "$OUTPUT_DIR/last.pt" || { echo "No last.pt to resume" >&2; exit 1; }
elif [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists. Choose a new directory or use RESUME=1." >&2
    exit 1
fi
# mkdir belongs to the allocated job, so a failed submission leaves no results folder.
raw_job=$(sbatch --parsable --export=ALL scripts/train_swin.slurm)
job_id="${raw_job%%;*}"
printf 'Job: %s\nResults: %s\n' "$job_id" "$OUTPUT_DIR"
printf 'Watch log: tail -F logs/swin_unet_%s.out\n' "$job_id"
printf 'Watch metrics: python scripts/monitor_epochs.py "%s" --watch\n' "$OUTPUT_DIR"
