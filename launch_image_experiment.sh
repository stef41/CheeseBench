#!/bin/bash
# Wait for current experiments (PID 1466246) to finish, then launch image mode experiment.

WAIT_PID=1466246
LOG="results/experiment_run_image.log"

echo "Waiting for PID $WAIT_PID to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
echo "PID $WAIT_PID finished at $(date)"

echo "Launching image mode experiment..."
python3 -u run_experiments.py --exp image_mode --num-trials 10 >> "$LOG" 2>&1
echo "Image mode experiment finished at $(date)" >> "$LOG"
echo "Done. Results in results/image_mode/"
