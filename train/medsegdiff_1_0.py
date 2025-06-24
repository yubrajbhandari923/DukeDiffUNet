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
from torch.optim.optimizer import Optimizer
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.engine import Events, Engine, EventEnum
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.metrics import Metric
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim


import monai
from monai.data import DataLoader, Dataset
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
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer, Inferer
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


from model.medsegdiffv2.guided_diffusion.resample import LossAwareSampler, UniformSampler
from model.medsegdiffv2.guided_diffusion.fp16_util import MixedPrecisionTrainer
from model.medsegdiffv2.guided_diffusion.utils import staple
from model.medsegdiffv2 import MedSegDiffModel
# from model.medsegdiffv2.guided_diffusion.custom_dataset_loader import CustomDataset3D

from utils.profiling import profile_block, TorchProfiler, init_profiler
from utils.monai_helpers import AimIgnite2DImageHandler

import subprocess

# region utils (EMA update, get_args, update_ema)

# def update_ema(target_params, source_params, rate=0.99):
#     """
#     Update target parameters to be closer to those of source parameters using
#     an exponential moving average.

#     :param target_params: the target parameter sequence.
#     :param source_params: the source parameter sequence.
#     :param rate: the EMA rate (closer to 1 means slower).
#     """
#     for targ, src in zip(target_params, source_params):
#         targ.detach().mul_(rate).add_(src, alpha=1 - rate)


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

# region Diffusion Trainer and Evaluator

class DiffusionTrainer(Trainer):
    """https://github.com/SuperMedIntel/MedSegDiff/blob/master/scripts/segmentation_train.py and TrainLoop in train_util.py"""
    def __init__(
        self,
        device: str | torch.device,
        max_epochs: int,
        train_data_loader: Iterable | DataLoader,
        network: torch.nn.Module,
        diffusion: Any,  # e.g. GaussianDiffusion
        optimizer: Optimizer,
        epoch_length: int | None = None,
        schedule_sampler: Any | None = None,  # e.g. UniformSampler
        prepare_batch: Callable = default_prepare_batch,
        inferer: Inferer | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Sequence | None = None,
        amp: bool = False,
        amp_kwargs: dict | None = None,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        to_kwargs: dict | None = None,
    ):
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            prepare_batch=prepare_batch,
            iteration_update=None,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )
        self.network = network
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.schedule_sampler = (
            schedule_sampler if schedule_sampler is not None else diffusion.uniform_sampler()
        )
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.network,
            use_fp16=False,
            fp16_scale_growth=self.amp_kwargs.get("fp16_scale_growth", 1e-3),
        )

    @profile_block("train_iteration")
    def _iteration(self, engine, batchdata):
        # logging.info(f"Running iteration on rank {engine.state.rank}")
        rank = engine.state.rank
        
        batchdata = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking
        )        
        images, labels = batchdata

        engine.network.train()
        self.mp_trainer.zero_grad()

        # For segmentation tasks, inputs are usually images and targets are segmentation masks.
        # But for diffusion models, segmentation masks are often used data and images are used as conditions.

        engine.state.output = {Keys.IMAGE: images, Keys.LABEL: labels}

        # in the original segDiff TrainLoop they have x_t from segmentation mask and {"conditioned_image": Image}

        # 2) sample timesteps & weights
        t, weights = self.schedule_sampler.sample(labels.shape[0], engine.state.device)

        # 3) compute the diffusion losses
        labels = torch.cat((labels, images), dim=1)  # Concatenate images and labels if needed

        # if rank == 0:
        # logging.info(f"Labels shape: {labels.shape}, t: {t.shape}, weights: {weights.shape}")
            
        with engine.network.no_sync():
            losses1 = self.diffusion.training_losses_segmentation(engine.network, None, labels, t, model_kwargs={})

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )

        losses = losses1[0]
        sample = losses1[1]

        loss = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()

        # store pred if your diffusion returns it
        engine.state.output[Keys.PRED] = sample
        engine.state.output[Keys.LOSS] = loss

        # 4) standard backward + step
        engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        self.mp_trainer.backward(loss)
        engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
        # engine.optimizer.step()
        self.mp_trainer.optimize(engine.optimizer)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output


class DiffusionEvaluator(Evaluator):

    """ Implement https://github.com/SuperMedIntel/MedSegDiff/blob/master/scripts/segmentation_sample.py """
    def __init__(
        self,
        device,
        val_data_loader,
        diffusion,
        network,
        schedule_sampler,
        prepare_batch=default_prepare_batch,
        n_rounds=1,
        major_vote_number=9,
        postprocessing=None,
        key_val_metric=None,
        val_handlers=None,
        amp=False,
        use_fp16=False,
        mode="eval",
        ddim=False,
    ):
        # Initialize base evaluator without default iteration_update
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=100,
            prepare_batch=prepare_batch,
            iteration_update=None,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=None,
            metric_cmp_fn= default_metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            decollate=False,
        )
        self.diffusion = diffusion
        self.network = network
        self.schedule_sampler = schedule_sampler
        self.n_rounds = n_rounds
        self.major_vote = major_vote_number
        self.ddim = ddim
        self.use_fp16 = use_fp16

    @profile_block("eval_iteration")
    def _iteration(self, engine, batchdata):
        # 1) prepare image and ground truth mask
        inputs, targets = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking
        )
        # engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}
        engine.state.output = {
            Keys.IMAGE: inputs,
            Keys.LABEL: targets,
            Keys.PRED: None,  # Placeholder for predictions
            'y': targets,  # Assuming 'y' is the ground truth mask
            'y_pred': None,  # Placeholder for predictions
        }

        batchsize = inputs.shape[0]

        # targets = torch.cat((targets, inputs), dim=1)  # Concatenate inputs and targets if needed
        noise = torch.randn_like(inputs[:, :1, ...])
        image = torch.cat((inputs, noise), dim=1)

        # if self.use_fp16:
        #     engine.network.convert_to_fp16()  # Convert model to half precision if using FP16

        # 2) perform major-vote sampling
        with torch.no_grad():
            # accumulate predictions
            votes = []
            for _ in range(self.n_rounds):
                # sample timesteps for each round
                # p_sample_loop will loop internally over timesteps

                # sample, x_noisy, org, cal, cal_out = sample_fn(
                #     model,
                #     (args.batch_size, 3, args.image_size, args.image_size),
                #     img,
                #     step=args.diffusion_steps,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=model_kwargs,
                # )

                if not self.ddim:
                    x = self.diffusion.p_sample_loop_known(
                        self.network,
                        (batchsize, 2, *inputs.shape[2:]), image,
                        # step = self.diffusion.num_timesteps,
                        step = 20, # Using DPMSolver++
                        noise=None,
                        clip_denoised=False,
                        denoised_fn=None,
                        model_kwargs={},
                        device=engine.state.device,
                        progress=False,
                    )
                else:
                    x = self.diffusion.ddim_sample_loop_known(
                        self.network,
                        (batchsize, 2, *inputs.shape[2:]), image,
                        step=self.diffusion.num_timesteps,
                        noise=None,
                        clip_denoised=False,
                        denoised_fn=None,
                        model_kwargs={},
                        device=engine.state.device,
                        progress=False,
                    )

                sample, x_noisy, org, cal, cal_out = x

                # co = cal_out
                votes.append(sample[:, -1, :, :])
                # torch.cuda.synchronize()

            # final vote: mean of all rounds
            final_vote = staple(torch.stack(votes, dim=0)).squeeze(0)

        # 3) store prediction
        # engine.state.output[Keys.PRED] = final_vote
        engine.state.output['y_pred'] = final_vote

        # 4) save as NIfTI
        # wrap in MetaTensor to preserve spatial metadata
        # SaveImage will read metadata from inputs
        engine.state.output[Keys.PRED] = final_vote

        # self.plot_image_label_pred(
        #     inputs, targets, final_vote,
        #     title=f"Rank {engine.state.rank}_Epoch_{engine.state.epoch}_Iteration_{engine.state.iteration}"
        # )

        # fire events for metrics
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output


# def train_step(engine, batch):
#     """
#     iteration_update function for training.
#     Expects engine.network, engine.diffusion, engine.optimizer,
#     engine.schedule_sampler, engine.mp_trainer, engine.optim_set_to_none
#     all to be set as attributes on the engine.
#     """
#     # 1) prepare
#     images, labels = engine.prepare_batch(
#         batch, engine.state.device, engine.non_blocking
#     )
#     engine.network.train()
#     engine.mp_trainer.zero_grad()

#     engine.state.output = {Keys.IMAGE: images, Keys.LABEL: labels}

#     # 2) sample timesteps & weights
#     t, weights = engine.schedule_sampler.sample(labels.shape[0], engine.state.device)

#     # 3) diffusion loss
#     combined = torch.cat((labels, images), dim=1)
#     with engine.network.no_sync():
#         loss_dict, sample = engine.diffusion.training_losses_segmentation(
#             engine.network, None, combined, t, model_kwargs={}
#         )
#     if isinstance(engine.schedule_sampler, LossAwareSampler):
#         engine.schedule_sampler.update_with_local_losses(t, loss_dict["loss"].detach())

#     loss = (loss_dict["loss"] * weights + loss_dict["loss_cal"] * 10).mean()
#     engine.state.output[Keys.PRED] = sample
#     engine.state.output[Keys.LOSS] = loss

#     # 4) backward + step
#     engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
#     engine.mp_trainer.backward(loss)
#     engine.mp_trainer.optimize(engine.optimizer)

#     return engine.state.output


# def eval_step(engine, batch):
#     """
#     iteration_update function for evaluation.
#     Expects engine.network, engine.diffusion, engine.schedule_sampler,
#     engine.n_rounds, engine.ddim all on the engine.
#     """
#     # 1) prepare
#     inputs, targets = engine.prepare_batch(
#         batch, engine.state.device, engine.non_blocking
#     )
#     engine.state.output = {
#         Keys.IMAGE: inputs,
#         Keys.LABEL: targets,
#         Keys.PRED: None,
#         "y": targets,
#         "y_pred": None,
#     }

#     batchsize = inputs.shape[0]
#     noise = torch.randn_like(inputs[:, :1, ...])
#     img_cond = torch.cat((inputs, noise), dim=1)

#     # 2) major‐vote sampling
#     votes = []
#     with torch.no_grad():
#         for _ in range(engine.n_rounds):
#             if not engine.ddim:
#                 out = engine.diffusion.p_sample_loop_known(
#                     engine.network,
#                     (batchsize, 2, *inputs.shape[2:]),
#                     img_cond,
#                     step=20,
#                     noise=None,
#                     clip_denoised=False,
#                     denoised_fn=None,
#                     model_kwargs={},
#                     device=engine.state.device,
#                     progress=False,
#                 )
#             else:
#                 out = engine.diffusion.ddim_sample_loop_known(
#                     engine.network,
#                     (batchsize, 2, *inputs.shape[2:]),
#                     img_cond,
#                     step=engine.diffusion.num_timesteps,
#                     noise=None,
#                     clip_denoised=False,
#                     denoised_fn=None,
#                     model_kwargs={},
#                     device=engine.state.device,
#                     progress=False,
#                 )
#             sample, *_ = out
#             votes.append(sample[:, -1, :, :])

#     final_vote = staple(torch.stack(votes, dim=0)).squeeze(0)
#     engine.state.output["y_pred"] = final_vote
#     engine.state.output[Keys.PRED] = final_vote

#     # fire metric events
#     engine.fire_event(IterationEvents.FORWARD_COMPLETED)
#     engine.fire_event(IterationEvents.MODEL_COMPLETED)
#     return engine.state.output


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
    # 1) prepare
    images, labels = engine.prepare_batch(
        batchdata, engine.state.device, engine.non_blocking
    )
    engine.network.train()
    engine.mp_trainer.zero_grad()

    # 2) sample timesteps & weights
    t, weights = engine.schedule_sampler.sample(labels.shape[0], engine.state.device)

    # 3) compute the diffusion loss
    combined = torch.cat((labels, images), dim=1)
    with engine.network.no_sync():
        loss_dict, sample = engine.diffusion.training_losses_segmentation(
            engine.network, None, combined, t, model_kwargs={}
        )
    if isinstance(engine.schedule_sampler, LossAwareSampler):
        engine.schedule_sampler.update_with_local_losses(t, loss_dict["loss"].detach())

    # weighted combination
    loss = (loss_dict["loss"] * weights + loss_dict["loss_cal"] * 10).mean()

    # 4) backward + step
    engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
    engine.mp_trainer.backward(loss)
    engine.mp_trainer.optimize(engine.optimizer)

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
    # 1) prepare
    inputs, targets = engine.prepare_batch(
        batchdata, engine.state.device, engine.non_blocking
    )

    # 2) noisy condition
    noise = torch.randn_like(inputs[:, :1, ...])
    cond = torch.cat((inputs, noise), dim=1)

    # 3) repeated sampling
    votes = []
    with torch.no_grad():
        for _ in range(engine.n_rounds):
            if not engine.ddim:
                out = engine.diffusion.p_sample_loop_known(
                    engine.network,
                    (inputs.shape[0], 2, *inputs.shape[2:]),
                    cond,
                    step=20,
                    noise=None,
                    clip_denoised=False,
                    model_kwargs={},
                    device=engine.state.device,
                    progress=False,
                )
            else:
                out = engine.diffusion.ddim_sample_loop_known(
                    engine.network,
                    (inputs.shape[0], 2, *inputs.shape[2:]),
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

    # 4) return a dict with keys "pred" and "label"
    #    MONAI's Evaluator will pick those up for metrics
    return {"image": inputs, "y_pred": pred, "y": targets}


# endregion


# region Logging and Config Handling (log_config, get_aim_logger, get_dataloaders, build_model, prepare_batch, get_metrics, get_post_processing)


def log_config(config, rank):
    if rank == 0:
        logging.info(f"Config: \n{OmegaConf.to_yaml(config)}")
        logging.info(f"MONAI version:  \n{monai.__version__}")
        logging.info(f"PyTorch version: \n{torch.__version__}")
        monai.config.print_config()


def get_aim_logger(config):

    if len(config.name) == 0:
        raise ValueError("Experiment name is required")

    aim_logger = AimLogger(
        repo=config.train.logging.aim_repo,
        experiment=f"{config.name}_{config.version}",
    )

    aim_logger.experiment.add_tag("Train")
    for tag in config.tags:
        aim_logger.experiment.add_tag(tag)

    # aim_logger.experiment.add_tag(config.name)

    aim_logger.experiment.description = config.description
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

class GetRandomSlice(monai.transforms.Transform):
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
            slice_index = np.random.choice(pos_indices)
        else:
            slice_index = np.random.choice(neg_indices)

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

def threshold_label(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).int()

@profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger, rank):
    train_files = []
    val_files = []

    # load the training and validation data
    with open(config.data.train_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            train_files.append({Keys.IMAGE: data["image"], Keys.LABEL: data["mask"]})
    train_files = train_files[:16]

    # train_files = train_files
    if aim_logger is not None and rank == 0:
        aim_logger.experiment.log_info(
            f"Training files {json.dumps(train_files, indent=2)}"
        )
        logging.info(f"Training files length: {len(train_files)}")

    with open(config.data.val_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            val_files.append({Keys.IMAGE: data["image"], Keys.LABEL: data["mask"]})
    val_files = val_files[:2]

    if aim_logger is not None and rank == 0:    
        aim_logger.experiment.log_info(
            f"Validation files {json.dumps(val_files, indent=2)}"
        )
        logging.info(f"Validation files length: {len(val_files)}")

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
             # Ensure the data is in the correct format
            # SplitDimd(
            #     keys=[Keys.IMAGE, Keys.LABEL],
            #     dim=-1,
            # ),
            GetRandomSlice(axis=config.data.slice_axis),  # Custom transform to get a random slice
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
            GetRandomSlice(axis=config.data.slice_axis),  # Custom transform to get a random slice
            ResizeWithPadOrCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.data.roi_size,
            ),
            Lambdad(keys=[Keys.LABEL], func=threshold_label),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = auto_dataloader(
        train_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = auto_dataloader(
        val_ds,
        batch_size=config.data.val_batch_size,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    return train_loader, val_loader


@profile_block("build_model")
def build_model(config, rank):
    # logging.info(f"{config.model}")
    segdiff = MedSegDiffModel(args=OmegaConf.to_container(config.model, resolve=True))
    net = segdiff.get_model().to(rank)

    opt = optim.AdamW(
        net.parameters(),
        config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )
    
    # dist.barrier()
    # net = DDP(net, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
    net = auto_model(net)
    opt = auto_optim(opt)

    if config.train.resume is not None:
        logging.info(f"[Rank {rank}] Loading model from {config.train.resume}")
        state_dict = torch.load(config.train.resume, map_location=f"cuda:{rank}")
        net.load_state_dict(state_dict["network"], strict=False)
        opt.load_state_dict(state_dict["optimizer"])
        epoch = state_dict.get("epoch", str(config.train.epochs).split(".")[0].split("_")[-1])
        
        logging.info(f"[Rank {rank}] Model loaded successfully")
    else:
        logging.info(f"[Rank {rank}] No pre-trained model to load, starting from scratch")

    # dist_util.sync_params(net.parameters())

    diffusion = segdiff.get_diffusion()
    schedule_sampler = segdiff.get_schedule_sampler(config.model.schedule_sampler, config.model.diffusion_steps)

    return net, opt, diffusion, schedule_sampler

def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[Keys.LABEL].to(device, non_blocking=non_blocking)
    
    # logging.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

def prepare_val_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch.get(Keys.LABEL).to(device, non_blocking=non_blocking)
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

def attach_checkpoint_handler(trainer, net, opt, ema_params, config, rank):
    # if rank != 0:
    #     return
    try:
        # logging.info(f"[Rank {rank}] Attaching checkpoint handler")
        ckpt_dir = os.path.join(config.train.save_dir, config.name.lower() , "checkpoints")
        # os.makedirs(ckpt_dir, exist_ok=True)
        # logging.info(f"[Rank {rank}] Checkpoint directory: {ckpt_dir}")
        
        checkpoint_handler = ModelCheckpoint(
            dirname=ckpt_dir,
            filename_prefix=config.name,
            n_saved=None,
            require_empty=False,
            create_dir=True,
            global_step_transform=lambda eng, _: eng.state.epoch,
            save_on_rank=0,
        )
        # logging.info(f"[Rank {rank}] Checkpoint handler created")
        trainer.add_event_handler(
            # Events.EPOCH_COMPLETED, checkpoint_handler, {"network": net, "optimizer": opt, "ema_params": ema_params}
            Events.EPOCH_COMPLETED(every=config.train.save_interval), checkpoint_handler, {"network": net, "optimizer": opt}
        )
        logging.info(f"[Rank {rank}] Checkpoint handler attached successfully")
        
    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed in attach_checkpoint_handler: {e}")


def attach_ema_update(trainer, net, ema_params, config):
    def update_(engine):

        # logging.info(f"[Rank {engine.state.rank}] Updating EMA parameters")

        for p_ema, p in zip(ema_params, net.parameters()):
            # update_ema(p_ema, p, rate=config.model.ema_rate)
            p_ema.mul_(config.model.ema_rate).add_(p, alpha=1 - config.model.ema_rate)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1), update_)


def attach_ema_validation(trainer, net, ema_params, val_evaluator, val_loader, config):
    def validate_with_ema(engine):
        # save original
        orig = {n: p.detach().clone() for n, p in net.named_parameters()}
        # copy EMA weights into the model
        for p, p_ema in zip(net.parameters(), ema_params):
            p.data.copy_(p_ema.data)
        # run validation
        val_evaluator.run()
        # restore original weights
        for n, p in net.named_parameters():
            p.data.copy_(orig[n])

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.train.eval.validation_interval),
        validate_with_ema,
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
        patience=config.early_stopping_patience,
        score_function=stopping_fn_from_metric("Mean Dice"),
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopper)


def attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader, rank):
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
        
        aim_logger.attach(
            val_evaluator,
            log_handler=AimIgnite2DImageHandler(
                "Prediction",
                output_transform=from_engine(["image", "y", "y_pred"]),
                global_step_transform=global_step_from_engine(trainer),
            ),
            # event_name=Events.ITERATION_COMPLETED(
            #     every=2 if (len(val_loader) // 4) == 0 else len(val_loader) // 4
            # ),
            event_name=Events.ITERATION_COMPLETED
        )
        
    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed during attach_aim_handlers: {e}")

def start_aim_ui_server(config, rank=0):
    """
    Start the Aim UI server if it is not already running.
    This function checks if the Aim UI server is running and starts it if not.
    """
    if rank != 0:
        return
    try:
        # Check if Aim UI server is already running
        aim_repo = config.train.logging.aim_repo
        if not os.path.exists(aim_repo):
            logging.info(f"Aim repository {aim_repo} does not exist. Creating it.")
            os.makedirs(aim_repo, exist_ok=True)
        
        # Check if the port is already in use
        port_check = os.system(f"lsof -i :43800")
        if port_check == 0:
            logging.info("Aim UI server is already running on port 43800.")
            return
        logging.info("Starting Aim UI server...")
        # os.system(f"aim up --port 43800 --repo {config.train.logging.aim_repo} &")
        aim_process = subprocess.Popen(
            ["aim", "up", "--port", "43800", "--repo", config.train.logging.aim_repo],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.info(f"Aim UI server started successfully with PID {aim_process.pid}.")
        
    except Exception as e:
        logging.error(f"Failed to start Aim UI server: {e}")
        logging.error(f"Please start the Aim UI server manually using 'aim up --port 43800 --repo {config.train.logging.aim_repo}'")


def attach_handlers(trainer, val_evaluator, net, opt, ema_params, config, aim_logger, val_loader, rank):
    """
    Attach various handlers to the trainer and evaluator.
    """
    if rank == 0:
        logging.info(f"[Rank {rank}] Attaching handlers")
    
    attach_checkpoint_handler(trainer, net, opt, ema_params, config, rank)
    attach_ema_update(trainer, net, ema_params, config)
    attach_ema_validation(trainer, net, ema_params, val_evaluator, val_loader, config)
    attach_stats_handlers(trainer, val_evaluator, config, rank)
    # attach_early_stopping(val_evaluator, trainer, config, rank)
    attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader, rank)
# endregion


def _distributed_run(rank, config):
    device = idist.device()                          # e.g. 'cuda:0'
    # rank = idist.get_rank()
    world_size = idist.get_world_size()

    logging.info(f"[Rank {rank}] Running on device: {device}, world size: {world_size}")

    # log_config(config, rank)
    # idist.seed_everything(config.train.seed)

    aim_logger = None
    if rank == 0:
        aim_logger = get_aim_logger(config)
        start_aim_ui_server(config, rank)
    init_profiler(config, aim_logger, rank)

    train_loader, val_loader = get_dataloaders(config, aim_logger, rank)

    net, opt, diffusion, schduler = build_model(config, rank)

    # ema_params = copy.deepcopy(list(net.parameters()))
    ema_params = [p.detach().clone() for p in net.parameters()]

    # trainer = Trainer(
    #     device=device,
    #     max_epochs=config.train.epochs,
    #     train_data_loader=train_loader,
    #     network=net,
    #     optimizer=opt,
    #     prepare_batch=prepare_batch,
    #     diffusion=diffusion,
    #     schedule_sampler=schduler,
    # )
    trainer = Trainer(
        device=device,
        max_epochs=config.train.epochs,
        data_loader=train_loader,
        prepare_batch=prepare_batch,
        iteration_update=train_step,
        # key_metric={'loss': None},             # or your dict of Metrics
        additional_metrics=None,
        metric_cmp_fn=default_metric_cmp_fn,
    )

    # 2) Attach the objects your step functions need
    trainer.network           = net
    trainer.optimizer         = opt
    trainer.diffusion         = diffusion
    trainer.schedule_sampler  = schduler
    trainer.mp_trainer        = MixedPrecisionTrainer(
                                model=net,
                                use_fp16=False,
                                fp16_scale_growth=config.train.amp_kwargs.fp16_scale_growth,
                                )
    trainer.optim_set_to_none = config.train.optim_set_to_none

    # post_pred = AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=2)
    post_pred = AsDiscreted(keys="pred", argmax=True, to_onehot=2)
    post_label = AsDiscreted(keys="label", to_onehot=2)

    metrics = {"Mean Dice": MeanDice(include_background=False)}

    # val_evaluator = Evaluator(
    #     device=device,
    #     val_data_loader=val_loader,
    #     diffusion=diffusion,
    #     network=net,
    #     prepare_batch= prepare_batch,
    #     schedule_sampler=schduler,
    #     postprocessing=post_pred,
    #     key_val_metric=metrics,
    #     ddim=config.model.ddim,
    # )
    # logging.info(f"[Rank {rank}] Validation evaluator created")

    val_evaluator = Evaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=prepare_batch,
        iteration_update=eval_step,
        postprocessing=[post_label, post_pred],
        key_val_metric=metrics,
    )

    # 4) Attach objects needed during eval
    val_evaluator.network          = net
    val_evaluator.diffusion        = diffusion
    val_evaluator.schedule_sampler = schduler
    val_evaluator.n_rounds         = config.model.n_rounds
    val_evaluator.ddim             = config.model.ddim

    attach_handlers(
        trainer,
        val_evaluator,
        net,
        opt,
        ema_params,
        config,
        aim_logger,
        val_loader,
        rank
    )
    # Add barrier to ensure all processes are ready before starting training
    idist.utils.barrier()

    if rank == 0:
        logging.info(f"[Rank {rank}] Starting training for {config.train.epochs} epochs")
        logging.info(f"[Rank {rank}] Training on {len(train_loader.dataset)} training samples")
        logging.info(f"[Rank {rank}] Validation on {len(val_loader.dataset)} validation samples")

    with TorchProfiler(subdir="trainer_run"):
        trainer.run()

if __name__ == "__main__":
    config = get_args()
    # world_size = torch.cuda.device_count()
    # rank = int(os.environ["LOCAL_RANK"])  # torchrun provides this env var
    # logging.info(f"Running on rank {rank} with world size {world_size}")
    # main(rank, world_size)
    
    with idist.Parallel(backend="nccl", nproc_per_node=torch.cuda.device_count()) as parallel:
        parallel.run(_distributed_run, config)  # :contentReference[oaicite:2]{index=2}
    # _distributed_run(config)
