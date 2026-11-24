#!/bin/bash
# Run experiments one after another (one training at a time: each gets the whole GPU).
#
#   tools/run_queue.sh strnadnet_audited_labels size_strnadnet_small ...
#
# Each name is a module in experiments/. Output goes to $STRNADI_WS/logs/<name>.txt; a finished
# experiment (all folds have oof.npz) is skipped by the trainer itself, so the queue can be re-run.
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$STRNADI_WS/logs"
for name in "$@"; do
    echo "$(date '+%F %T') start $name" >> "$STRNADI_WS/logs/queue.txt"
    python -m "experiments.$name" > "$STRNADI_WS/logs/$name.txt" 2>&1
    echo "$(date '+%F %T') done  $name (exit $?)" >> "$STRNADI_WS/logs/queue.txt"
done
