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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.engine import Events, Engine, EventEnum
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.metrics import Metric

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
)
from monai.engines.utils import (
    IterationEvents,
    default_prepare_batch,
    default_metric_cmp_fn,
)
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


# region utils (EMA update, get_args, update_ema)

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


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
        n_rounds=3,
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
            metric_cmp_fn=None,
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
                        step = self.diffusion.num_timesteps,
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

                co = torch.tensor(cal_out)
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

        # fire events for metrics
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output


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

    aim_logger.experiment.add_tag(config.name)

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
    """
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        
    def __call__(self, data):
        image = data[Keys.IMAGE]
        label = data[Keys.LABEL]
        
        if image.ndim != 4 or label.ndim != 4:
            raise ValueError("Input images must be 4D tensors ((batch), channel, depth, height, width)")

        # Get a random slice index
        slice_index = np.random.randint(image.shape[self.axis])
        
        # Select the slice
        if self.axis == 1 or self.axis == -3:
            image_slice = image[:, slice_index, :, :]
            label_slice = label[:, slice_index, :, :]
        elif self.axis == 2 or self.axis == -2:
            image_slice = image[:, :, slice_index, :]
            label_slice = label[:, :, slice_index, :]
        elif self.axis == 3 or self.axis == -1:
            image_slice = image[:, :, :, slice_index]
            label_slice = label[:, :, :, slice_index]
        else:
            raise ValueError("Axis must be 1, 2, or 3 (or -3, -2, -1 for negative indexing)")
        
        return {Keys.IMAGE: image_slice, Keys.LABEL: label_slice}

@profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger, rank, world_size):
    train_files = []
    val_files = []

    # load the training and validation data
    with open(config.data.train_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            train_files.append({Keys.IMAGE: data["image"], Keys.LABEL: data["mask"]})
    # train_files = train_files[:2]

    train_files = train_files
    if aim_logger is not None and rank == 0:
        aim_logger.experiment.log_info(
            f"Training files {json.dumps(train_files, indent=2)}"
        )
        logging.info(f"Training files length: {len(train_files)}")

    
    with open(config.data.val_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            val_files.append({Keys.IMAGE: data["image"], Keys.LABEL: data["mask"]})
    # val_files = val_files[:2]

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
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@profile_block("build_model")
def build_model(config, rank):
    # logging.info(f"{config.model}")
    segdiff = MedSegDiffModel(args=OmegaConf.to_container(config.model, resolve=True))
    net = segdiff.get_model().to(rank)

    dist.barrier()
    net = DDP(net, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)

    if config.train.resume is not None:
        logging.info(f"[Rank {rank}] Loading model from {config.train.resume}")
        state_dict = torch.load(config.train.resume, map_location=rank)
        net.load_state_dict(state_dict["model"], strict=False)
        logging.info(f"[Rank {rank}] Model loaded successfully")
    else:
        logging.info(f"[Rank {rank}] No pre-trained model to load, starting from scratch")

    # dist_util.sync_params(net.parameters())

    diffusion = segdiff.get_diffusion()
    schedule_sampler = segdiff.get_schedule_sampler(config.model.schedule_sampler, config.model.diffusion_steps)

    opt = optim.AdamW(
        net.parameters(),
        config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )
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
            Events.EPOCH_COMPLETED, checkpoint_handler, {"network": net, "optimizer": opt}
        )
        logging.info(f"[Rank {rank}] Checkpoint handler attached successfully")
        
    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed in attach_checkpoint_handler: {e}")


def attach_ema_update(trainer, net, ema_params, config):
    def update_(engine):
        
        # logging.info(f"[Rank {engine.state.rank}] Updating EMA parameters")
        
        for p_ema, p in zip(ema_params[0], net.parameters()):
            update_ema(p_ema, p, rate=config.model.ema_rate)
    

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1), update_)


def attach_ema_validation(trainer, net, ema_params, val_evaluator, val_loader, config):
    def validate_with_ema(engine):
        original_state = copy.deepcopy(net.state_dict())
        for p, p_ema in zip(net.parameters(), ema_params[0]):
            p.data.copy_(p_ema.data)
        val_evaluator.run()
        net.load_state_dict(original_state)

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
    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed during attach_aim_handlers: {e}")

# endregion

def main(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

    config = get_args()

    config.train.device = f"cuda:{rank}"

    init_profiler(config)

    log_config(config, rank)
    torch.random.manual_seed(config.train.seed)

    if rank == 0:
        try:
            aim_logger = get_aim_logger(config)
        except Exception as e:
            logging.exception("[Rank 0] Failed to initialize Aim logger")
            aim_logger = None
    else:
        aim_logger = None

    train_loader, val_loader = get_dataloaders(config, aim_logger, rank, world_size)

    net, opt, diffusion, schduler = build_model(config, rank)

    ema_params = [copy.deepcopy(list(net.parameters())) for _ in range(1)]

    trainer = DiffusionTrainer(
        device=config.train.device,
        max_epochs=config.train.epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        prepare_batch=prepare_batch,
        diffusion=diffusion,
        schedule_sampler=schduler,
    )

    post_pred = AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=2)
    metrics = {"Mean Dice": MeanDice(include_background=False)}

    val_evaluator = DiffusionEvaluator(
        device=config.train.device,
        val_data_loader=val_loader,
        diffusion=diffusion,
        network=net,
        prepare_batch= prepare_batch,
        schedule_sampler=schduler,
        postprocessing=post_pred,
        key_val_metric=metrics,
        ddim=config.model.ddim,
    )
    logging.info(f"[Rank {rank}] Validation evaluator created")

    attach_checkpoint_handler(trainer, net, opt, ema_params, config, rank)

    logging.info(f"[Rank {rank}] Checkpoint handler attached")

    # attach_stats_handlers(trainer, val_evaluator, config, rank)

    attach_ema_update(trainer, net, ema_params, config)
    logging.info(f"[Rank {rank}] EMA update handler attached")

    # attach_ema_validation(trainer, net, ema_params, val_evaluator, val_loader, config)

    # attach_validation(trainer, val_evaluator, config)

    # attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader, rank)

    # val_evaluator.run()

    # Add barrier to ensure all processes are ready before starting training
    dist.barrier()
    
    if rank == 0:
        logging.info(f"[Rank {rank}] Starting training for {config.train.epochs} epochs")
        logging.info(f"[Rank {rank}] Training on {len(train_loader.dataset)} training samples")
        logging.info(f"[Rank {rank}] Validation on {len(val_loader.dataset)} validation samples")
    with TorchProfiler(subdir="trainer_run"):
        trainer.run()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    rank = int(os.environ["LOCAL_RANK"])  # torchrun provides this env var
    logging.info(f"Running on rank {rank} with world size {world_size}")
    main(rank, world_size)
