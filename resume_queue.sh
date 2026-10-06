#!/usr/bin/env bash
# resume_queue.sh — restart the experiment queue after a power cut
#
#   bash resume_queue.sh            # resume everything
#   bash resume_queue.sh --status   # progress only

cd "$(dirname "$0")" || exit 1
source ordinalfed_env/Scripts/activate
python experiments/run_queue.py "$@"
