#!/usr/bin/env bash
# Run one training session: start the run if it does not exist yet, otherwise
# resume it from last.pth.  Run it again later to continue; it exits cleanly
# after EPOCHS (or HOURS), on Ctrl+C, or when you `touch logs/<dataset>/<name>/STOP`.
#
#   tools/train_session.sh                      # defaults below
#   EPOCHS=2 tools/train_session.sh             # shorter session
#   HOURS=6 EPOCHS=0 tools/train_session.sh     # stop on time instead
#   tools/train_session.sh --lr 2e-5            # extra args go to train.py
#
# Every setting that shapes the run (NUM_EPOCH, BATCH, ...) must stay the same
# for all sessions of one NAME; train.py warns if it changes.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-vilnm_bs8_e30}
DATASET=${DATASET:-CUHK-PEDES}
NUM_EPOCH=${NUM_EPOCH:-30}
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-8}
EVAL_PERIOD=${EVAL_PERIOD:-5}
EPOCHS=${EPOCHS:-4}          # epochs per session (0 = no limit)
HOURS=${HOURS:-0}            # hours per session (0 = no limit)
PYTHON=${PYTHON:-.venv/bin/python}

# conda's activate hook points LD_LIBRARY_PATH at its own CUDA libraries, which
# breaks the venv's cuDNN (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED).
unset LD_LIBRARY_PATH

LAST="logs/${DATASET}/${NAME}/last.pth"
RESUME=()
if [[ -f "$LAST" ]]; then
    RESUME=(--resume "$LAST")
    echo ">> resuming ${NAME} from ${LAST}"
else
    echo ">> starting new run ${NAME}"
fi

# systemd-inhibit keeps the machine from suspending while the session runs.
INHIBIT=()
if command -v systemd-inhibit >/dev/null; then
    INHIBIT=(systemd-inhibit --what=sleep:idle --who=ViLNM --why="training ${NAME}")
fi

exec "${INHIBIT[@]}" "$PYTHON" train.py \
    --dataset_name "$DATASET" --name "$NAME" \
    --num_epoch "$NUM_EPOCH" --batch_size "$BATCH" --amp True \
    --num_workers "$WORKERS" --eval_period "$EVAL_PERIOD" \
    --stop_after_epochs "$EPOCHS" --max_hours "$HOURS" \
    --mlflow True "${RESUME[@]}" "$@"
