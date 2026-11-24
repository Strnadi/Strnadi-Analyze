#!/bin/bash
# Full evaluation suite for a finished CV run. Run from the repository root.
# usage: evaluation/research_session/post_eval.sh <run> [cores, e.g. 43-44]
run=$1; cores=${2:-}
WS=${STRNADI_WS:-$HOME/strnadi-ml}
[ -f "$WS/env.sh" ] && source "$WS/env.sh"
export THREADS=2 OMP_NUM_THREADS=2
PY="python -W ignore -m"; [ -n "$cores" ] && PY="taskset -c $cores $PY"
OUT=$WS/results/post_$run.txt
{
echo "=== $run $(date)"
$PY evaluation.report $run
echo "--- Vizina held-out males (fold ensemble). NOTE: not an independent test for runs trained on Vizina/JCU weak data (neighbouring males, same territories)"
$PY evaluation.research_session.eval_external torch:$WS/runs/$run $WS/data/vizina_eval.csv test 2>&1 | tail -2
echo "--- expert BE/BlBh birds (out of fold, songs re-detected)"; $PY evaluation.research_session.eval_expert_birds $run 2>&1 | tail -1
echo "--- bird-level NCS recordings (out of fold)"; $PY evaluation.eval_recordings $run 2>&1 | head -1
[ -f $WS/data/jcu_test.csv ] && { echo "--- JCU held-out recorders (rewards a 'passive => BhBl' shortcut; read with the next line)"; $PY evaluation.research_session.eval_constrained $run $WS/data/jcu_test.csv 2>&1 | tail -1; }
[ -f $WS/data/farfield_stats.csv ] && { echo "--- dialect prior shift under passive-recorder acoustics"; $PY evaluation.research_session.dialect_prior_shift $run 2>&1 | tail -1; }
} > $OUT 2>&1
cat $OUT
