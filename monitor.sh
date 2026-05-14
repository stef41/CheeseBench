#!/bin/bash
# Monitor CheeseBench experiment progress
echo "=== CheeseBench Experiment Monitor ==="
echo "Time: $(date)"
echo ""

# Check if experiment is running
PID=$(cat /tmp/experiment_pid.txt 2>/dev/null)
if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
    echo "✓ Experiment running (PID: $PID)"
else
    echo "✗ Experiment NOT running"
fi
echo ""

# Check GPU
echo "GPU Memory (MiB):"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -2
echo ""

# Check completed results
echo "Completed experiments:"
find results/ -name benchmark_results.json 2>/dev/null | sort | while read f; do
    SIZE=$(stat --format=%s "$f" 2>/dev/null || echo "?")
    echo "  ✓ $f ($SIZE bytes)"
done
echo ""

# Check what's running now
echo "Current progress (from experiment log):"
tail -5 results/experiment_run.log 2>/dev/null
echo ""

# Check current model's trace
LATEST_TRACE=$(find results/ -name llm_traces.log -newer /tmp/experiment_pid.txt 2>/dev/null | tail -1)
if [ -n "$LATEST_TRACE" ]; then
    echo "Latest trace: $LATEST_TRACE"
    echo "  Environments reached:"
    grep "^Environment:" "$LATEST_TRACE" 2>/dev/null | tail -5
    echo "  Trials completed: $(grep -c '^Result:' "$LATEST_TRACE" 2>/dev/null)"
    echo "  Last results:"
    grep "^Result:" "$LATEST_TRACE" 2>/dev/null | tail -3
fi
