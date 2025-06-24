#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

# python -m train.segdiff_1_0 --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/segdiff.yaml
pipenv run bash -c "CUDA_VISIBLE_DEVICES=7 nohup torchrun --master-port=29712 --nproc_per_node=1 -m train.medsegdiff_1_0 \
  --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff_inference.yaml > /home/yb107/logs/val_medsegdiff.log 2>&1 &"

echo "Started torchrun (PID $$) — logs: /home/yb107/logs/val_medsegdiff.log"
echo "To stop the process, use: kill $$"
echo "To view logs, use: tail -f /home/yb107/logs/val_medsegdiff.log"
echo "To view the process list, use: ps aux | grep torchrun"
echo "To check GPU usage, use: nvidia-smi"