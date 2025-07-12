import os
import sys
import logging
import json
import copy
import monai.transforms
import numpy as np
import argparse
import functools
import nibabel

from typing import Iterable, Callable, Any, Sequence
import aim
from aim.pytorch_ignite import AimLogger
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from ignite.engine import Events, Engine, EventEnum
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim
from torch.optim.swa_utils import AveragedModel

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Spacingd,
    EnsureChannelFirstd,
    AsDiscreted,
    ResizeWithPadOrCropd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    SplitDimd,
    Lambdad,
)
from monai.engines.utils import (
    IterationEvents,
    default_prepare_batch,
    default_metric_cmp_fn,
)
from monai.handlers import (
    MeanDice,
    StatsHandler,
    IgniteMetricHandler,
    stopping_fn_from_metric,
    from_engine,
)
from monai.data import list_data_collate, NumpyReader
from monai.engines import Evaluator, Trainer
from monai.utils.enums import CommonKeys as Keys
from monai.data.utils import pickle_hashing

from model.medsegdiffv2.guided_diffusion.resample import LossAwareSampler, UniformSampler
from model.medsegdiffv2.guided_diffusion.utils import staple
from model.medsegdiffv2 import MedSegDiffModel
# from model.medsegdiffv2.guided_diffusion.custom_dataset_loader import CustomDataset3D

from utils.profiling import profile_block, TorchProfiler, init_profiler
from utils.monai_helpers import AimIgnite2DImageHandler

import subprocess
import torch.multiprocessing as tmp_mp

tmp_mp.set_sharing_strategy("file_system")

DEBUG = False
# region utils (EMA update, get_args, update_ema)

def get_args():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument(
        "--base_config",
        type=str,
        default="/home/yb107/cvpr2025/DukeDiffSeg/configs/base.yaml",
        help="Path to base config file (default: configs/base.yaml)",
    )

    parser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        help="Path to experiment config file that overrides base",
    )

    args = parser.parse_args()

    # Load and merge configs
    base_cfg = OmegaConf.load(args.base_config)
    exp_cfg = OmegaConf.load(args.exp_config)
    config = OmegaConf.merge(base_cfg, exp_cfg)

    return config

# endregion

# region train_step and eval_step:

def train_step(engine, batchdata):
    """
    This will replace your DiffusionTrainer._iteration.
    Expects these attributes on `engine`:
      - engine.network
      - engine.diffusion
      - engine.schedule_sampler
      - engine.optimizer
      - engine.mp_trainer
      - engine.optim_set_to_none (bool)
    """
    # logging.info(
    #     f"Train [Rank {idist.get_rank()}] Starting step at iteration {engine.state.iteration}"
    # )

    # 1) prepare
    images, labels = engine.prepare_batch(
        batchdata, engine.state.device, engine.non_blocking
    )
    engine.network.train()

    optimizer = engine.optimizer
    scaler = engine.scaler_ 
    use_amp = engine.amp

    # 2) sample timesteps & weights
    t, weights = engine.schedule_sampler.sample(labels.shape[0], engine.state.device)

    optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

    # 3) compute the diffusion loss
    # combined = torch.cat((labels, images), dim=1)
    combined = torch.cat((images, labels), dim=1)

    with autocast(device_type='cuda', enabled=use_amp):
        loss_dict, sample = engine.diffusion.training_losses_segmentation(
            engine.network, None, combined, t, model_kwargs={}
        )

    if isinstance(engine.schedule_sampler, LossAwareSampler):
        engine.schedule_sampler.update_with_local_losses(t, loss_dict["loss"].detach())
        
    loss = (loss_dict["loss"] * weights + loss_dict["loss_cal"] * 10).mean()

    
    # 4) backward + step
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    for name, param in engine.network.named_parameters():
        if param.grad is None:
            logging.warning(f"Parameter {name} has no gradient. This may indicate an issue with the model or data.")
            
    # 5) return whatever you need in metrics/handlers
    #    here we return a dict so StatsHandler can pick out "loss"
    return {"loss": loss, "pred": sample, "label": labels}


def eval_step(engine, batchdata):
    """
    This replaces your DiffusionEvaluator._iteration.
    Expects these attrs on `engine`:
      - engine.network
      - engine.diffusion
      - engine.schedule_sampler
      - engine.n_rounds
      - engine.ddim
    """
    # logging.info(
    #     f"Eval [Rank {idist.get_rank()}] Starting step at iteration {engine.state.iteration}"
    # )

    # 1) prepare
    image, masks = engine.prepare_batch(
        batchdata, engine.state.device, engine.non_blocking
    )
    config = engine.config

    # 2) noisy condition
    noise = torch.randn_like(image[:, :1, ...])
    cond = torch.cat((image, noise), dim=1)

    # 3) repeated sampling
    votes = []
    with torch.no_grad():
        for _ in range(engine.n_rounds):
            if not engine.ddim:
                out = engine.diffusion.p_sample_loop_known(
                    engine.network,
                    (image.shape[0], 2, *image.shape[2:]),
                    cond,
                    step=config.diffusion.diffusion_steps,
                    noise=None,
                    clip_denoised=config.diffusion.clip_denoised,
                    model_kwargs={},
                    device=engine.state.device,
                    progress=False,
                )
            else:
                out = engine.diffusion.ddim_sample_loop_known(
                    engine.network,
                    (image.shape[0], 2, *image.shape[2:]),
                    cond,
                    step=engine.diffusion.num_timesteps,
                    noise=None,
                    clip_denoised=False,
                    model_kwargs={},
                    device=engine.state.device,
                    progress=False,
                )
            sample, *_ = out
            votes.append(sample[:, -1, :, :])

    pred = staple(torch.stack(votes, dim=0)).squeeze(0)

    pred = pred.unsqueeze(1)  # Add channel dimension
    pred = (pred > 0.5).to(torch.int32)
    # logging.info(f"Images shape: {image.shape}, Labels shape: {masks.shape}, Predictions shape: {pred.shape}")

    return {"image": image, "y_pred": pred, "y": masks}

# endregion

# region Logging and Config Handling :


def log_config(config, rank):
    if rank == 0:
        logging.info(f"Config: \n{OmegaConf.to_yaml(config)}")
        logging.info(f"MONAI version:  \n{monai.__version__}")
        logging.info(f"PyTorch version: \n{torch.__version__}")
        monai.config.print_config()


def get_aim_logger(config):

    if not config.experiment.name or len(config.experiment.name) == 0:
        raise ValueError("Experiment name is required")

    aim_logger = AimLogger(
        repo=config.logging.aim_repo,
        experiment=f"{config.experiment.name}_{config.experiment.version}",
    )

    aim_logger.experiment.add_tag("Train")
    for tag in config.experiment.tags:
        aim_logger.experiment.add_tag(tag)

    # aim_logger.experiment.add_tag(config.name)

    aim_logger.experiment.description = config.experiment.description
    aim_logger.log_params(OmegaConf.to_container(config, resolve=True))
    aim_logger.experiment.log_info(
        OmegaConf.to_yaml(config),
    )

    # Log this script's content
    script_path = os.path.abspath(__file__)
    with open(script_path, "r") as script_file:
        script_content = script_file.read()
    aim_logger.experiment.log_info(script_content)

    return aim_logger


# endregion

# region Custom Transforms

class GetRandomSlice(monai.transforms.RandomizableTransform):
    """
    A custom transform to get a random slice from a 3D image.
    pos and neg are the proportions of samples with and without labels present.
    axis is the axis along which to slice the image.
    """
    def __init__(self, axis, pos=10, neg=1):
        super().__init__()
        self.axis = axis
        self.pos = pos
        self.neg = neg

        if self.axis == -1:
            self.axis = 3
        elif self.axis == -2:
            self.axis = 2
        elif self.axis == -3:
            self.axis = 1

        if self.axis not in [1, 2, 3]:
            raise ValueError("Axis must be 1 (depth), 2 (height), or 3 (width) for 4D tensors or -1, -2, -3 for negative indexing.")

    def __call__(self, data):
        image = data[Keys.IMAGE]
        label = data[Keys.LABEL]

        if image.ndim != 4 or label.ndim != 4:
            raise ValueError("Input images must be 4D tensors ((batch), channel, depth, height, width)")
        # Get the shape of the image
        channels, depth, height, width = image.shape        

        # Randomly select a slice index along the specified axis
        if self.axis == 1:  # Slicing along the depth axis
            # Get the slices with and without labels present
            pos_indices = np.where(label.sum(axis=(0, 2, 3)) > 0)[0]
            neg_indices = np.where(label.sum(axis=(0, 2, 3)) == 0)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                logging.warning(f"No positive or negative samples found in {image.meta['filename_or_obj']}. Using all slices.")
                pos_indices = np.arange(depth)
                neg_indices = np.arange(depth)

        elif self.axis == 2:  # Slicing along the height axis
            # Get the slices with and without labels present
            pos_indices = np.where(label.sum(axis=(0, 1, 3)) > 0)[0]
            neg_indices = np.where(label.sum(axis=(0, 1, 3)) == 0)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                logging.warning(
                    f"No positive or negative samples found in  {image.meta['filename_or_obj']}. Using all slices."
                )
                pos_indices = np.arange(height)
                neg_indices = np.arange(height)

        elif self.axis == 3:  # Slicing along the width axis
            # Get the slices with and without labels present
            pos_indices = np.where(label.sum(axis=(0, 1, 2)) > 0)[0]
            neg_indices = np.where(label.sum(axis=(0, 1, 2)) == 0)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                logging.warning(f"No positive or negative samples found in  {image.meta['filename_or_obj']} Using all slices.")
                pos_indices = np.arange(width)
                neg_indices = np.arange(width)
        else:
            raise ValueError("Invalid axis specified. Must be 0, 1, or 2.")

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            # raise ValueError("No positive or negative samples found in the dataset")
            logging.warning("No positive or negative samples found in the dataset. Using all slices.")

        # Randomly select a slice index with a probability based on pos and neg
        if np.random.rand() < (self.pos / (self.pos + self.neg)):
            slice_index = self.R.choice(pos_indices)
        else:
            slice_index = self.R.choice(neg_indices)

        # Create slices for the image and label
        if self.axis == 1:  # Slicing along the depth axis
            image_slice = image[:, slice_index, :, :]
            label_slice = label[:, slice_index, :, :]
        elif self.axis == 2:  # Slicing along the height axis
            image_slice = image[:, :, slice_index, :]
            label_slice = label[:, :, slice_index, :]
        elif self.axis == 3:  # Slicing along the width axis
            image_slice = image[:, :, :, slice_index]
            label_slice = label[:, :, :, slice_index]
        else:
            raise ValueError("Invalid axis specified. Must be 0, 1, or 2.")

        return {Keys.IMAGE: image_slice, Keys.LABEL: label_slice}


class GetSliced(monai.transforms.MapTransform):
    def __init__(self, keys, slice_key="slice_idx", axis=-1):
        super().__init__(keys)
        self.axis = axis
        self.slice_key = slice_key
        
        if self.axis == -1:
            self.axis = 3
        elif self.axis == -2:
            self.axis = 2
        elif self.axis == -3:
            self.axis = 1

        if self.axis not in [1, 2, 3]:
            raise ValueError("Axis must be 1 (depth), 2 (height), or 3 (width) for 4D tensors or -1, -2, -3 for negative indexing.")
    
    def __call__(self, data):
        """
        Slices the input data along the specified axis.
        """
        d = dict(data)
        slice_idx = d.get(self.slice_key)
        
        for key in self.keys:
            if key not in d:
                raise KeyError(f"Key '{key}' not found in input data.")
            image = d[key]
            if image.ndim != 4:
                raise ValueError(f"Input images must be 4D tensors ((batch), channel, depth, height, width) for key '{key}'")

            # Create slices for the image
            if self.axis == 1:
                # Slicing along the depth axis
                d[key] = image[:, slice_idx, :, :]
            elif self.axis == 2:
                # Slicing along the height axis
                d[key] = image[:, :, slice_idx, :]
            elif self.axis == 3:
                # Slicing along the width axis
                d[key] = image[:, :, :, slice_idx]
                
        return d

# endregion

# region Data Loading and Preprocessing

def threshold_label(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).int()

def volume_hash(item):
    # keep only the fields you care about
    filtered = {
        Keys.IMAGE: item[Keys.IMAGE],
        Keys.LABEL: item[Keys.LABEL],
    }
    # use MONAI's pickle_hashing on the filtered dict
    return pickle_hashing(filtered)


@profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger, rank):
    train_files = []
    val_files = []

    with open(config.data.train_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            train_files.append({Keys.IMAGE: data["image"], Keys.LABEL: data["mask"]})

    with open(config.data.val_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            val_files.append({Keys.IMAGE: data["image"], Keys.LABEL: data["mask"]})

    if rank == 0:
        aim_logger.experiment.log_info(
            f"Training files {json.dumps(train_files, indent=2)}"
        )
        logging.info(f"Training files length: {len(train_files)}")
        aim_logger.experiment.log_info(
            f"Validation files {json.dumps(val_files, indent=2)}"
        )
        logging.info(f"Validation files length: {len(val_files)}")
       
    if DEBUG:
        train_files = train_files[:10]
        val_files = val_files[:10]

    if len(val_files) > config.evaluation.validation_max_num_samples:
        # Randomly sample validation slices to limit the number of samples
        logging.info(f"Validation files length before sampling: {len(val_files)}")
        np.random.seed(config.seed)
        np.random.shuffle(val_files)
        val_files = val_files[:config.evaluation.validation_max_num_samples]
        logging.info(f"Validation files length after sampling: {len(val_files)}")

    train_files = json.dumps(train_files)
    val_files = json.dumps(val_files)


    train_transforms = Compose(
        [
            LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            Orientationd(
                keys=[Keys.IMAGE, Keys.LABEL],
                axcodes=config.data.orientation,
            ),
            Lambdad(keys=[Keys.LABEL], func=threshold_label),
  
            RandSpatialCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                roi_size=config.data.roi_size,
                random_size=False,
            ),
            ResizeWithPadOrCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.data.roi_size,
            ),
            ToTensord(keys=[Keys.IMAGE, Keys.LABEL]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            Orientationd(
                keys=[Keys.IMAGE, Keys.LABEL],
                axcodes=config.data.orientation,
            ),
            Lambdad(keys=[Keys.LABEL], func=threshold_label),
            
            ResizeWithPadOrCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.data.roi_size,
            ),
        ]
    )
    # create a training data loader
    train_ds = monai.data.CacheNTransDataset(
        data=train_files, transform=train_transforms, cache_n_trans=5, cache_dir=config.data.cache_dir, hash_func=volume_hash,
    )
    train_loader = auto_dataloader(
        train_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=True,
    )
    # create a validation data loader
    val_ds = monai.data.CacheNTransDataset(
        data=val_files,
        transform=val_transforms,
        cache_dir=config.data.cache_dir,
        cache_n_trans=5,
        hash_func=volume_hash,
    )
    val_loader = auto_dataloader(
        val_ds,
        batch_size=config.data.val_batch_size,
        num_workers=config.data.val_num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    return train_loader, val_loader


@profile_block("build_model")
def build_model(config, rank):
    segdiff = MedSegDiffModel(model_config=config.model.params, diffusion_config=config.diffusion)
    net = segdiff.get_model().to(rank)

    if config.optimizer.name == "AdamW":
        opt = optim.AdamW(
            net.parameters(),
            config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")

    net = auto_model(net)
    opt = auto_optim(opt)
    # ema_params = [p.detach().clone() for p in net.parameters()]
    ema_model = AveragedModel(
        net,
        avg_fn=lambda avg_p, new_p, _: avg_p.mul_(config.model.params.ema_rate).add_(
            new_p, alpha=1 - config.model.params.ema_rate
        ),
    )

    resume_epoch = 0
    if config.training.resume is not None:
        state_dict = torch.load(config.training.resume, map_location=f"cuda:{rank}")

        net.load_state_dict(state_dict["network"], strict=False)
        opt.load_state_dict(state_dict["optimizer"])
        ema_model.load_state_dict(state_dict.get("ema_model", ema_model.state_dict()))

        resume_epoch = str(config.training.resume).split(".")[0].split("_")[-1]
        if "epoch" in state_dict:
            resume_epoch = state_dict["epoch"]

        if isinstance(resume_epoch, str):
            resume_epoch = int(resume_epoch)
        logging.info(f"[Rank {rank}] Resuming from epoch {resume_epoch}")
        logging.info(f"[Rank {rank}] Model loaded successfully")
    else:
        logging.info(f"[Rank {rank}] No pre-trained model to load, starting from scratch")

    # dist_util.sync_params(net.parameters())

    diffusion = segdiff.get_diffusion()
    schedule_sampler = segdiff.get_schedule_sampler(config.diffusion.schedule_sampler.name, config.diffusion.schedule_sampler.max_steps)

    return net, opt, diffusion, schedule_sampler, ema_model, resume_epoch

def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[Keys.LABEL].to(device, non_blocking=non_blocking)
    
    # logging.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

def get_metrics():
    # create a dictionary of metrics
    metrics = {
        "Mean Dice": MeanDice(
            include_background=False,
            # to_onehot_y=True, softmax=True, batch=True
        ),
    }
    return metrics

def get_post_processing():
    # create a post-processing transform
    post_pred = Compose(
        [
            AsDiscreted(
                keys=Keys.LABEL, to_onehot=2
            ),  # Assuming 2 classes for binary segmentation
            AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=2),
        ]
    )
    return post_pred

# endregion

# region Handlers

def attach_checkpoint_handler(trainer, net, opt, ema_model, config, rank):
    if rank != 0:
        return
    try:
        ckpt_dir = os.path.join(config.training.save_dir, config.experiment.name.lower() , "checkpoints")
        
        checkpoint_handler = ModelCheckpoint(
            dirname=ckpt_dir,
            filename_prefix=config.experiment.name,
            n_saved=10,
            require_empty=False,
            create_dir=True,
            global_step_transform=lambda eng, _: eng.state.epoch,
            save_on_rank=0,
        )
        latest_ckpt = ModelCheckpoint(
            dirname=ckpt_dir,
            filename_prefix=f"{config.experiment.name}_latest",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            global_step_transform=lambda eng, _: eng.state.epoch,
            save_on_rank=0,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config.training.save_interval), checkpoint_handler, {"network": net, "optimizer": opt, "ema_model": ema_model}
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=1), latest_ckpt, {"network": net, "optimizer": opt, "ema_model": ema_model}
        )
        logging.info(f"[Rank {rank}] Checkpoint handler attached successfully")
        
    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed in attach_checkpoint_handler: {e}")


def attach_ema_update(trainer, net, ema_model, config):
    # def update_(engine):

    #     # logging.info(f"[Rank {engine.state.rank}] Updating EMA parameters")

    #     for p_ema, p in zip(ema_params, net.parameters()):
    #         # update_ema(p_ema, p, rate=config.model.ema_rate)
    #         p_ema.mul_(config.model.ema_rate).add_(p, alpha=1 - config.model.ema_rate)

    def update_ema(engine):
        ema_model.update_parameters(engine.network)  # updates the buffers
        # logging.info(f"[EMA] updated at iter {engine.state.iteration}")

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1), update_ema)


def attach_ema_validation(trainer, net, ema_model, val_evaluator, val_loader, config):
    # def validate_with_ema(engine):
    #     # save original
    #     orig = {n: p.detach().clone() for n, p in net.named_parameters()}
    #     # copy EMA weights into the model
    #     for p, p_ema in zip(net.parameters(), ema_params):
    #         p.data.copy_(p_ema.data)
    #     # run validation
    #     val_evaluator.run()
    #     # restore original weights
    #     for n, p in net.named_parameters():
    #         p.data.copy_(orig[n])

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.evaluation.validation_interval),
        lambda engine: val_evaluator.run()
    )


def attach_stats_handlers(trainer, val_evaluator, config, rank):
    if rank != 0:
        return
    
    StatsHandler(
        name="trainer",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
    ).attach(trainer)
    StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
        iteration_log=False,
    ).attach(val_evaluator)


def attach_early_stopping(val_evaluator, trainer, config, rank):
    if rank != 0:
        return
    stopper = EarlyStopping(
        patience=config.evaluation.early_stopping.patience,
        score_function=stopping_fn_from_metric("Mean Dice"),
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopper)


def attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader, rank, config):
    if rank != 0 or aim_logger is None:
        return
    try:
        # Output loss
        for tag, event in [
            ("Iteration Dice Loss", Events.ITERATION_COMPLETED),
            ("Epoch Dice Loss", Events.EPOCH_COMPLETED),
        ]:
            aim_logger.attach_output_handler(
                trainer,
                event_name=event,
                tag=tag,
                output_transform=from_engine(["loss"], first=True),
            )

        aim_logger.attach_output_handler(
            val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="val",
            metric_names=["Mean Dice"],
            global_step_transform=global_step_from_engine(trainer),
        )

        if config.evaluation.visualize:
            aim_logger.attach(
                val_evaluator,
                log_handler=AimIgnite2DImageHandler(
                    "Prediction",
                    output_transform=from_engine(["image", "y", "y_pred"]),
                    global_step_transform=global_step_from_engine(trainer),
                ),
                event_name=Events.ITERATION_COMPLETED(
                    every=config.evaluation.visualize_every_iter
                ),
            )
    except Exception as e:
       logging.exception(f"[Rank {rank}] Failed during attach_aim_handlers: {e}")

def attach_handlers(trainer, val_evaluator, net, opt, ema_model, config, aim_logger, val_loader, rank):
    """
    Attach various handlers to the trainer and evaluator.
    """
    if rank == 0:
        logging.info(f"[Rank {rank}] Attaching handlers")
    
    attach_checkpoint_handler(trainer, net, opt, ema_model, config, rank)
    
    attach_ema_update(trainer, net, ema_model, config)
    attach_ema_validation(trainer, net, ema_model, val_evaluator, val_loader, config)
    
    attach_stats_handlers(trainer, val_evaluator, config, rank)
    # attach_early_stopping(val_evaluator, trainer, config, rank)
    attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader, rank, config)
# endregion


def _distributed_run(rank, config):
    device = idist.device()                          # e.g. 'cuda:0'
    world_size = idist.get_world_size()

    logging.info(f"[Rank {rank}] Running on device: {device}, world size: {world_size}")

    # log_config(config, rank)
    # idist.seed_everything(config.train.seed)

    aim_logger = None
    if rank == 0:
        aim_logger = get_aim_logger(config)
    init_profiler(config, aim_logger, rank)

    train_loader, val_loader = get_dataloaders(config, aim_logger, rank)

    net, opt, diffusion, scheduler, ema_model, resume_epoch = build_model(config, rank)

    trainer = Trainer(
        device=device,
        max_epochs=config.training.epochs,
        data_loader=train_loader,
        prepare_batch=prepare_batch,
        iteration_update=train_step,
        # key_metric={'loss': None},             # or your dict of Metrics
        additional_metrics=None,
        metric_cmp_fn=default_metric_cmp_fn,
        amp=config.amp.enabled,
    )

    # 2) Attach the objects your step functions need
    trainer.network           = net
    trainer.optimizer         = opt
    trainer.diffusion         = diffusion
    trainer.schedule_sampler  = scheduler
    trainer.scaler_             = GradScaler(enabled=config.amp.enabled)

    trainer.optim_set_to_none = config.optimizer.set_to_none

    # post_pred = AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=2)
    post_pred = AsDiscreted(keys="y_pred", threshold=0.5, to_onehot=2)
    # post_pred = AsDiscreted(keys="y_pred", argmax=True, to_onehot=2)
    post_label = AsDiscreted(keys="y", to_onehot=2)

    metrics = {"Mean Dice": MeanDice(include_background=False)}

    val_evaluator = Evaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=prepare_batch,
        iteration_update=eval_step,
        postprocessing=Compose([post_label, post_pred]),
        key_val_metric=metrics,
    )

    # 4) Attach objects needed during eval
    val_evaluator.network          = ema_model
    # val_evaluator.network          = net
    val_evaluator.diffusion        = diffusion
    val_evaluator.schedule_sampler = scheduler
    val_evaluator.n_rounds         = config.model.params.n_rounds
    val_evaluator.ddim             = config.diffusion.ddim
    val_evaluator.config           = config

    attach_handlers(
        trainer,
        val_evaluator,
        net,
        opt,
        ema_model,
        config,
        aim_logger,
        val_loader,
        rank
    )
    # Add barrier to ensure all processes are ready before starting training
    idist.utils.barrier()

    if rank == 0:
        logging.info(f"[Rank {rank}] Starting training for {config.training.epochs} epochs")
        logging.info(f"[Rank {rank}] Training on {len(train_loader.dataset)} training samples")
        logging.info(f"[Rank {rank}] Validation on {len(val_loader.dataset)} validation samples")

    with TorchProfiler(subdir="trainer_run"):
        trainer.state.epoch = resume_epoch
        if resume_epoch > 0:
            logging.info(f"[Rank {rank}] Resuming training from epoch {resume_epoch}")
        trainer.run()

if __name__ == "__main__":
    config = get_args()
    # world_size = torch.cuda.device_count()
    # rank = int(os.environ["LOCAL_RANK"])  # torchrun provides this env var
    # logging.info(f"Running on rank {rank} with world size {world_size}")
    # main(rank, world_size)
    
    with idist.Parallel(backend="nccl", nproc_per_node=torch.cuda.device_count()) as parallel:
        parallel.run(_distributed_run, config)  
