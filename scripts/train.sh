#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

# python -m train.segdiff_1_0 --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/segdiff.yaml
pipenv run bash -c "CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=8 nohup torchrun --nproc_per_node=4 -m train.medsegdiff_1_0 \
  --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff.yaml > /home/yb107/logs/train_medsegdiff.log 2>&1 &"

echo "Started torchrun — logs: /home/yb107/logs/train_medsegdiff.log"
echo "To stop the process, use: 'ps aux | grep torchrun && kill <PID>'"
echo "To view logs, use: tail -f /home/yb107/logs/train_medsegdiff.log"
echo "To check GPU usage, use: nvidia-smi"
