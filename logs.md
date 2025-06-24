## 🧪 Experiment Log – 2025-06-04 6:20PM: 
Running the Basic DDP code on plp-capri.

CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=8 nohup torchrun --nproc_per_node=4 -m train.segdiff_1_0 \
  --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/segdiff.yaml > /home/yb107/logs/train_segdiff.log 2>&1 &
  (Because 1,2 GPU are running in capri for some reason)
  
- Look at `tail -f /home/yb107/logs/train_segdiff.log`
- To kill the nohup process: `ps aux | grep train.segdiff_1_0` then `kill -9 $pid`
OR `ps aux | grep torchrun`

For some reason rank 0 seems not to be working.

Type log <Tab> to log stuff

## 🧪 Experiment Log – 2025-06-23 5:21 PM

Medsegdiff ran for 200 epochs. need to fix eval code to make it run quickly.

## 🧪 Experiment Log – 2025-06-24 11:12 AM

Tried running inference on the trained code. Prediction was all zeros. Need to dig into the actual diffsion code base.

Meeting Notes:
Why this method? 
- Porbably is better (especially in the hard cases like diseased liver/organs or tumors). Can prove this by running on the quality control failed cases of Dukeseg
- Uncertainity based method
- 

