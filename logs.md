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

## 🧪 Experiment Log – 2025-06-27 1:50 PM

So, Done with refining the codebase.
At this point should be eaiser to bring in any dataloaders and models and quickly switch between the models.

Questions right now:
- On the aim logs, loss is > 1, why ? shouldn't it have been dice or something. Look in the Paper to understand this
- Also, the results showed basically all zeros generating.
- 

## 🧪 Experiment Log – 2025-07-03

🔹 Purpose of Schedule Sampler
The purpose of a schedule sampler is to:
 - Control the distribution of timesteps sampled during training.
- Improve training efficiency by prioritizing timesteps where the model is underperforming or where gradients are more informative.
 - Balance learning across the full diffusion trajectory — from clean images (t = 0) to highly noised images (t = T).

Without a smart sampler, the model may waste training capacity on timesteps that don’t need much improvement.

## 🧪 Experiment Log – 2025-07-22

Whats Next for SPIE ?
- Look at the results of colon segmentation. If its great voila. Most likely wont. Go back to generative task - Kauther's Problem. Constraint generation of Colon using diffusion model.
- When should I use loss scalar ? Should I always use it ?
- Run nnUNET to compare colon segementation results with
- low res then high res generation for whole abdomen

## 🧪 Experiment Log – 2025-07-24
So just looking at the validation dice score between nnUNet and DiffUnet, unnet had 0.9 ish both for multi organs and colon only, while diffunet had 0.6 ish for colon only and 0.4 ish for multi organ, so its not really an improvement. But since the diffunet were not converged completely, and I messed up by not saving lr_scheduler's state the training has restarted. But I am doubtful if it will be on par as nnUNet because 0.9 is extremely good.

What next?
Generative approach, fix the colon by generating the missing parts, and constrained by surrounding organs.

Chatgpt: 
        https://chatgpt.com/share/68827b8b-037c-8011-b2a8-9fa8c6dad665

Need following organs for constraining colon generation

Small intestine (SI)	
Liver	
Spleen	
Stomach	
Kidneys	
Pancreas	
Bladder	
Uterus/prostate 
vertebrae	
Duodenum

Todo: (Data Analysis)
  - Get colon, and add few centimeters up and down, and  see what organs are present in few 100 cases then decide on the organs to include.
  - Think about the loss function.
  - 

Idea: Use Signed Distance Transforms to enforce spatial awareness, which is what those organ masks are supposed to do anyway.
Idea: Low res -> super resolution generation
Idea: Latent Diffusion ?

Idea: 
      Some sort of topological loss (Euler characteristic difference, surface distance)
      Shape plausibility (PCA on shape descriptors or Procrustes distance to mean colon shape)
      Smoothness & connectivity (Hausdorff distance, boundary continuity)

Idea: 
    Generate both large and small intestine and compare with trying to generate just large intestine


## 🧪 Experiment Log – 2025-07-25
Goal: Generate Anatomically realistic colon masks constrained by surrounding organs.
Experiments to Run:
  DiffUNet 2.0: Generate Colon segmentation mask conditioned on binary contrains
  DiffUNet 2.1 : Generate Colon Segmentation mask conditioned on multi-class one hot constraints
  
  DiffUNet 2.2: Generate Colon and Small Bowel conditioned on binary constraints
  DiffUNet 2.3: Generate Colon and Small Bowel conditioned on multi-class constraints
  
  DiffUNet 2.4: Generate Colon conditioned on multi-class SDF constraints
  DiffUNet 2.5: Generate Colon and Small Bowel conditioned on SDF constraints

  DiffUNet 2.6: Generate Colon SDF conditioned on binary constraints
  DiffUNet 2.7: Generate Colon SDF conditioned on multi-class constraints

  Other experiments: Try different losses, like incorporate weighted topological loss on early epochs and decrease losses, or weighted Atlas based shape prior loss, or topological loss.
  the code already contains Dice+BCE+MSE.


