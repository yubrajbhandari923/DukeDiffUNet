#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

LOGFILE="/home/yb107/logs/val_medsegdiff.log"
PIDFILE="/home/yb107/logs/val_medsegdiff.pid"

# Clean up any old PID file
if [ -f "$PIDFILE" ]; then
    rm "$PIDFILE"
fi

# Launch in new process group with setsid
pipenv run bash -c "CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 setsid nohup python -m inference.medsegdiff_1_0 \
  --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff_inference.yaml > $LOGFILE 2>&1 & echo \$! > $PIDFILE"

echo "🚀 Training started — logs: $LOGFILE"
echo "📄 Main PID saved to: $PIDFILE"
echo "🔍 View logs: tail -f $LOGFILE"
echo "🛑 Kill training and all subprocesses: /home/yb107/cvpr2025/DukeDiffSeg/scripts/kill_training.sh"
echo "🎮 Check GPU usage: nvidia-smi"

