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
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim
from torch.optim.swa_utils import AveragedModel
from ignite.utils import setup_logger


import monai
from monai import transforms
from monai.engines.utils import default_metric_cmp_fn, IterationEvents
from monai.handlers import (
    MeanDice,
    StatsHandler,
    stopping_fn_from_metric,
    from_engine,
)
from monai.data import list_data_collate, NumpyReader
from monai.engines import Evaluator, Trainer
from monai.utils.enums import CommonKeys as Keys
from monai.data.utils import pickle_hashing
from monai.losses.dice import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.apps import get_logger

# from model.medsegdiffv2.guided_diffusion.resample import LossAwareSampler, UniformSampler
# from model.medsegdiffv2.guided_diffusion.utils import staple
# from model.medsegdiffv2 import MedSegDiffModel
# from model.medsegdiffv2.guided_diffusion.custom_dataset_loader import CustomDataset3D

from model.diffUNet.BTCV import DiffUNet

from utils.profiling import profile_block, TorchProfiler, init_profiler
from utils.monai_helpers import (
    AimIgnite2DImageHandler,
    AimIgniteGIFHandler,
    AimIgnite3DImageHandler,
)
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

import subprocess
import torch.multiprocessing as tmp_mp

tmp_mp.set_sharing_strategy("file_system")
torch.serialization.add_safe_globals([monai.utils.enums.CommonKeys])
# stash the original loader
_torch_load = torch.load

# override so all loads are unguarded
torch.load = lambda f, **kwargs: _torch_load(f, weights_only=False, **kwargs)

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

    setup_logger(
        name="training_logger",  # this is the logger that StatsHandler will use
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s: %(message)s",
        reset=True,  # drop any old handlers on this logger
    )
    return config

# endregion

# region train_step and eval_step:
def train_step(engine, batchdata):
    optimizer = engine.optimizer
    scaler = engine.scaler_
    use_amp = engine.amp
    config = engine.config
    accum = config.training.accumulate_grad_steps

    # 1) prepare
    images, labels = engine.prepare_batch(
        batchdata, engine.state.device, engine.non_blocking
    )
    images = images.float()
    labels = labels.float()
    # labels_int = labels.long().squeeze(1)  # Assuming labels are in shape [B, 1, D, H, W]

    # labels_1hot = (
    #     nn.functional.one_hot(labels_int, num_classes=config.data.num_classes)
    #     .permute(0, 4, 1, 2, 3)
    #     .float()
    # )

    labels_1hot = labels

    engine.network.train()

    if engine.state.iteration == 1:
        optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

    # 3) Train Step
    x_start = (labels_1hot) * 2 - 1
    x_t, t, noise = engine.network(x=x_start, pred_type="q_sample")
    pred_xstart = engine.network(x=x_t, step=t, image=images, pred_type="denoise")
    engine.fire_event(IterationEvents.FORWARD_COMPLETED)

    # labels_1hot = labels
    loss_dice = engine.dice_loss(pred_xstart, labels_1hot)
    loss_bce = engine.bce(pred_xstart, labels_1hot)

    pred_xstart = torch.sigmoid(pred_xstart)
    # pred_xstart = torch.softmax(pred_xstart, dim=1)

    loss_mse = engine.mse(pred_xstart, labels_1hot)

    loss = loss_dice + loss_bce + loss_mse

    loss = loss / accum
    engine.fire_event(IterationEvents.LOSS_COMPLETED)

    # backward
    scaler.scale(loss).backward() if use_amp else loss.backward()

    # optimizer step on the last micro-step
    if engine.state.iteration % accum == 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        for name, param in engine.network.named_parameters():
            if param.grad is None:
                logging.warning(
                    f"Parameter {name} has no gradient. This may indicate an issue with the model or data."
                )
                
        optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

    engine.fire_event(IterationEvents.MODEL_COMPLETED)
    return {"loss": loss * accum, "label": labels}


def eval_step(engine, batchdata):
    # 1) prepare
    image, masks = engine.prepare_batch(
        batchdata, engine.state.device, engine.non_blocking
    )
    image = image.float()
    masks = masks.float()

    engine.state.output = {Keys.IMAGE: image, Keys.LABEL: masks}

    config = engine.config

    engine.network.eval()

    with engine.mode(engine.network):
        model_fn = lambda x: engine.network(image=x, pred_type="ddim_sample")
        if engine.amp:
            with torch.autocast("cuda", **engine.amp_kwargs):
                engine.state.output[Keys.PRED] = engine.inferer(
                    image, engine.network, pred_type="ddim_sample"
                )
        else:
            # engine.state.output[Keys.PRED] = engine.inferer(image, engine.network, pred_type="ddim_sample")
            engine.state.output[Keys.PRED] = engine.inferer(image, model_fn)

    # engine.state.output[Keys.PRED] = (engine.state.output[Keys.PRED] + 1) / 2.0
    engine.state.output[Keys.PRED] = torch.sigmoid(engine.state.output[Keys.PRED])

    if DEBUG:
        raw_logits = engine.state.output[Keys.PRED]
        logging.info(
            f"Logits stats: min={raw_logits.min():.4f}, max={raw_logits.max():.4f}, mean={raw_logits.mean():.4f}"
        )

        # Log Shapes and Unique Values
        logging.info(f"Output shape: {engine.state.output[Keys.PRED].shape}")
        logging.info(f"Image shape: {image.shape}")
        logging.info(f"Masks shape: {masks.shape}")

    engine.fire_event(IterationEvents.FORWARD_COMPLETED)
    engine.fire_event(IterationEvents.MODEL_COMPLETED)

    return engine.state.output


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

# region Data Loading and Preprocessing


def threshold_label(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Remove values greater than num_classes from the label tensor."""
    x[x >= num_classes] = 0
    return x


def mask_label(x: torch.Tensor, label: int) -> torch.Tensor:
    """Mask out the specified label in the label tensor."""
    x[x != label] = 0
    x[x == label] = 1
    return x


def custom_name_formatter(meta_dict, saver):
    full_path = meta_dict["filename_or_obj"]
    base = os.path.basename(full_path)
    # If the filename itself contains "colon", pull the parent folder as the ID

    if "labels" in full_path.lower():
        postfix = "_label"
    else:
        # strip off ".nii.gz" (7 chars) or any extension
        postfix = "_image"
    # saver.file_postfix is either "_image" or "_label"
    return {"filename": f"{base.replace('.nii.gz', '')}{postfix}"}


@profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger, rank):
    train_files = []
    val_files = []

    with open(config.data.train_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            train_files.append(
                {
                    Keys.IMAGE: data["image"],
                    Keys.LABEL: data["mask"],
                }
            )

    with open(config.data.val_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            val_files.append(
                {
                    Keys.IMAGE: data["image"],
                    Keys.LABEL: data["mask"],
                }
            )

    if rank == 0:
        aim_logger.experiment.track(
            aim.Text(f"{json.dumps(train_files, indent=2)}"), name="Training Files", step=1
        )
        logging.info(f"Training files length: {len(train_files)}")
        aim_logger.experiment.track(
            aim.Text(f"{json.dumps(val_files, indent=2)}"), name="Validation Files", step=1
        )
        logging.info(f"Validation files length: {len(val_files)}")

    if DEBUG:
        train_files = train_files[:12]
        val_files = val_files[:4]

    if len(val_files) > config.evaluation.validation_max_num_samples:
        # Randomly sample validation slices to limit the number of samples
        logging.info(f"Validation files length before sampling: {len(val_files)}")
        np.random.seed(config.seed)
        np.random.shuffle(val_files)
        val_files = val_files[: config.evaluation.validation_max_num_samples]
        logging.info(f"Validation files length after sampling: {len(val_files)}")

    # TODO: Crop foreground for training and validation abdomenal region
    # TODO: Train seperate model using different Data.jsonl
    # TODO: Predict 14 classes and only colon on new data

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            transforms.EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            transforms.Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            transforms.Orientationd(
                keys=[Keys.IMAGE, Keys.LABEL],
                axcodes=config.data.orientation,
            ),
            transforms.ScaleIntensityRanged(
                keys=[Keys.IMAGE],
                a_min=-175,
                a_max=250.0,
                b_min=0,
                b_max=1.0,
                clip=True,
            ),
            transforms.CropForegroundd(
                keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.IMAGE
            ),
            transforms.SpatialPadd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.data.roi_size,
                mode=["constant", "constant"],
                constant_values=[0, 0],
            ),
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=functools.partial(
                    threshold_label, num_classes=config.data.num_classes
                ),
            ),
            transforms.RandCropByPosNegLabeld(
                keys=[Keys.IMAGE, Keys.LABEL],
                label_key=Keys.LABEL,
                spatial_size=(96, 96, 96),
                pos=5,
                neg=1,
                num_samples=4 if not DEBUG else 1,
                image_key=Keys.IMAGE,
                image_threshold=0,
            ),
            # transforms.RandFlipd(keys=[Keys.IMAGE, Keys.LABEL], prob=0.2, spatial_axis=0),
            # transforms.RandFlipd(keys=[Keys.IMAGE, Keys.LABEL], prob=0.2, spatial_axis=1),
            # transforms.RandFlipd(keys=[Keys.IMAGE, Keys.LABEL], prob=0.2, spatial_axis=2),
            # transforms.RandRotate90d(keys=[Keys.IMAGE, Keys.LABEL], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys=Keys.IMAGE, factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys=Keys.IMAGE, offsets=0.1, prob=0.1),
            # (
            #     transforms.SaveImaged(
            #         keys=[Keys.IMAGE, Keys.LABEL],
            #         meta_keys=[f"{Keys.IMAGE}_meta_dict", f"{Keys.LABEL}_meta_dict"],
            #         output_dir=os.path.join(
            #             config.training.save_dir,
            #             config.experiment.name.lower(),
            #             "training_samples",
            #         ),
            #         output_postfix="",
            #         separate_folder=False,
            #         output_name_formatter=custom_name_formatter,
            #     )
            #     if DEBUG
            #     else transforms.Identityd(keys=[Keys.IMAGE, Keys.LABEL])
            # ),
            transforms.AsDiscreted(
                keys=[Keys.LABEL],
                to_onehot=config.data.num_classes,
            ),
            transforms.ToTensord(
                keys=[Keys.IMAGE, Keys.LABEL],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            # lambda data: (
            #     logging.info(
            #         f"Preprocess data"
            #     ),
            #     data,
            # )[1],
            transforms.LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            transforms.EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            transforms.Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            transforms.Orientationd(
                keys=[Keys.IMAGE, Keys.LABEL],
                axcodes=config.data.orientation,
            ),
            transforms.ScaleIntensityRanged(
                keys=[Keys.IMAGE],
                a_min=-175,
                a_max=250.0,
                b_min=0,
                b_max=1.0,
                clip=True,
            ),
            transforms.CropForegroundd(
                keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.IMAGE
            ),
            (
                transforms.CropForegroundd(
                    keys=[Keys.IMAGE, Keys.LABEL],
                    source_key=Keys.LABEL,
                )
                if DEBUG
                else transforms.Identityd(keys=[Keys.IMAGE, Keys.LABEL])
            ),
            transforms.SpatialPadd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.data.roi_size,
                mode=["constant", "constant"],
                constant_values=[0, 0],
            ),
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=functools.partial(
                    threshold_label, num_classes=config.data.num_classes
                ),
            ),
            # (
            #     transforms.SaveImaged(
            #         keys=[Keys.IMAGE, Keys.LABEL],
            #         meta_keys=[f"{Keys.IMAGE}_meta_dict", f"{Keys.LABEL}_meta_dict"],
            #         output_dir=os.path.join(
            #             config.training.save_dir,
            #             config.experiment.name.lower(),
            #             "validation_samples",
            #         ),
            #         output_postfix="",
            #         separate_folder=False,
            #         output_name_formatter=custom_name_formatter,
            #     )
            #     if DEBUG
            #     else transforms.Identityd(keys=[Keys.IMAGE, Keys.LABEL])
            # ),
            transforms.AsDiscreted(
                keys=[Keys.LABEL],
                to_onehot=config.data.num_classes,
            ),
            transforms.ToTensord(keys=[Keys.IMAGE, Keys.LABEL]),
        ]
    )
    # create a training data loader
    train_ds = monai.data.PersistentDataset(
        data=train_files,
        transform=train_transform,
        cache_dir=config.data.cache_dir,
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
    val_ds = monai.data.PersistentDataset(
        data=val_files,
        transform=val_transform,
        cache_dir=config.data.cache_dir,
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
    net = DiffUNet(
        spatial_dims=config.model.params.spatial_dims,
        in_channels=config.model.params.in_channels,
        out_channels=config.model.params.out_channels,
        features=config.model.params.features,
        # activation=tuple(config.model.params.activation),
        # dropout_rate=config.model.params.dropout_rate,
        # use_checkpointing=config.model.params.use_checkpointing,
        diffusion_steps=config.diffusion.diffusion_steps,
        beta_schedule=config.diffusion.beta_schedule,
        ddim_steps=config.diffusion.ddim_steps,
        image_size=config.data.roi_size,
    )

    if config.optimizer.name == "AdamW":
        opt = optim.AdamW(
            net.parameters(),
            config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")

    lr_scheduler = None
    if config.lr_scheduler.name == "LinearWarmupCosineAnnealingLR":
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=config.lr_scheduler.warmup_epochs,
            max_epochs=config.lr_scheduler.max_epochs,
        )

    scaler = GradScaler(enabled=config.amp.enabled)

    if rank == 0:
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info(f"[Rank {rank}] Model parameters: {num_params / 1e6:.2f}M")

    net = auto_model(net)
    opt = auto_optim(opt)
    # ema_params = [p.detach().clone() for p in net.parameters()]
    ema_model = AveragedModel(
        net,
        avg_fn=lambda avg_p, new_p, _: avg_p.mul_(config.ema.ema_rate).add_(
            new_p, alpha=1 - config.ema.ema_rate
        ),
    )

    resume_epoch = 0
    if config.training.resume is not None:
        state_dict = torch.load(config.training.resume, map_location=f"cuda:{rank}")

        net.load_state_dict(state_dict["network"], strict=False)
        opt.load_state_dict(state_dict["optimizer"])
        ema_model.load_state_dict(state_dict.get("ema_model", ema_model.state_dict()))
        if "lr_scheduler" in state_dict:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        else:
            logging.warning(
                f"[Rank {rank}] No lr_scheduler found in checkpoint, using default."
            )

        if "scaler" in state_dict:
            scaler.load_state_dict(state_dict["scaler"])
        else:
            logging.warning(
                f"[Rank {rank}] No scaler found in checkpoint, using default."
            )

        resume_epoch = str(config.training.resume).split(".")[0].split("_")[-1]
        if "epoch" in state_dict:
            resume_epoch = state_dict["epoch"]
       

        if isinstance(resume_epoch, str):
            resume_epoch = int(resume_epoch)
        logging.info(f"[Rank {rank}] Resuming from epoch {resume_epoch}")
        logging.info(f"[Rank {rank}] Model loaded successfully")
    else:
        logging.info(
            f"[Rank {rank}] No pre-trained model to load, starting from scratch"
        )

    # dist_util.sync_params(net.parameters())

    # diffusion = segdiff.get_diffusion()
    # schedule_sampler = segdiff.get_schedule_sampler(config.diffusion.schedule_sampler.name, config.diffusion.schedule_sampler.max_steps)

    return net, opt, lr_scheduler, ema_model, resume_epoch, scaler


def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[Keys.LABEL].to(device, non_blocking=non_blocking)

    # logging.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels


# endregion

# region Handlers


def attach_checkpoint_handler(
    trainer, val_evaluator, net, opt, lr_scheduler, scaler, ema_model, config, rank
):
    if rank != 0:
        return
    try:
        ckpt_dir = os.path.join(
            config.training.save_dir, config.experiment.name.lower(), "checkpoints"
        )

        best_metric_checkpoint_handler = ModelCheckpoint(
            dirname=ckpt_dir,
            filename_prefix=f"{config.experiment.name}_best",
            n_saved=10,
            score_name="Mean Dice",
            require_empty=False,
            create_dir=True,
            global_step_transform=global_step_from_engine(trainer),
            save_on_rank=0,
        )
        latest_ckpt = ModelCheckpoint(
            dirname=ckpt_dir,
            filename_prefix=f"{config.experiment.name}_latest",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            global_step_transform=global_step_from_engine(trainer),
            save_on_rank=0,
        )
        val_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            best_metric_checkpoint_handler,
            {"network": net, "optimizer": opt, "ema_model": ema_model, "lr_scheduler": lr_scheduler, "scaler": scaler},
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=1),
            latest_ckpt,
            {"network": net, "optimizer": opt, "ema_model": ema_model, "lr_scheduler": lr_scheduler, "scaler": scaler},
        )
        logging.info(f"[Rank {rank}] Checkpoint handler attached successfully")

    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed in attach_checkpoint_handler: {e}")


def attach_ema_update(trainer, net, ema_model, config):
    def update_ema(engine):
        ema_model.update_parameters(engine.network)  # updates the buffers

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
        lambda engine: val_evaluator.run(),
    )


def attach_stats_handlers(trainer, val_evaluator, config, rank):
    if rank != 0:
        return

    StatsHandler(
        name="training_logger",
        output_transform=from_engine(["loss"], first=True),
        global_epoch_transform=lambda epoch : trainer.state.epoch,
        iteration_log=False,
        tag_name="Dice Loss",
    ).attach(trainer)
    
    StatsHandler(
        name="training_logger",
        output_transform=lambda x: None,
        global_epoch_transform=lambda epoch: trainer.state.epoch,
        iteration_log=False,
        tag_name="Dice Metric",
    ).attach(val_evaluator)


def attach_early_stopping(val_evaluator, trainer, config, rank):
    if rank != 0:
        return
    if not config.evaluation.early_stopping.enabled:
        logging.info(f"[Rank {rank}] Early stopping is disabled")
        return
    stopper = EarlyStopping(
        patience=config.evaluation.early_stopping.patience,
        score_function=stopping_fn_from_metric("Mean Dice"),
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopper)
    logging.info(f"[Rank {rank}] Early stopping attached")


def attach_aim_handlers(
    trainer, val_evaluator, aim_logger, val_loader, rank, config, postprocess=None
):
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
                log_handler=AimIgnite3DImageHandler(
                    "Prediction",
                    output_transform=from_engine([Keys.IMAGE, Keys.LABEL, Keys.PRED]),
                    global_step_transform=global_step_from_engine(trainer),
                    postprocess=postprocess,
                ),
                event_name=Events.ITERATION_COMPLETED(
                    every=1 if DEBUG else config.evaluation.visualize_every_iter
                ),
            )
    except Exception as e:
        logging.exception(f"[Rank {rank}] Failed during attach_aim_handlers: {e}")


def attach_handlers(
    trainer,
    val_evaluator,
    net,
    opt,
    lr_scheduler,
    scaler,
    ema_model,
    config,
    aim_logger,
    val_loader,
    rank,
    postprocess=None,
):
    """
    Attach various handlers to the trainer and evaluator.
    """
    if rank == 0:
        logging.info(f"[Rank {rank}] Attaching handlers")

    attach_checkpoint_handler(trainer, val_evaluator, net, opt, lr_scheduler, scaler, ema_model, config, rank)

    attach_ema_update(trainer, net, ema_model, config)
    attach_ema_validation(trainer, net, ema_model, val_evaluator, val_loader, config)

    attach_stats_handlers(trainer, val_evaluator, config, rank)
    attach_early_stopping(val_evaluator, trainer, config, rank)
    attach_aim_handlers(
        trainer,
        val_evaluator,
        aim_logger,
        val_loader,
        rank,
        config,
        postprocess=postprocess,
    )

    # Step lr_scheduler at the end of each epoch
    if config.lr_scheduler.name == "LinearWarmupCosineAnnealingLR":
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, lambda engine: engine.lr_scheduler.step()
        )

    trainer.add_event_handler(
        Events.STARTED,
        lambda engine: engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none),
    )


# endregion


def _distributed_run(rank, config):
    device = idist.device()  # e.g. 'cuda:0'
    world_size = idist.get_world_size()

    logging.info(f"[Rank {rank}] Running on device: {device}, world size: {world_size}")

    aim_logger = None
    if rank == 0:
        aim_logger = get_aim_logger(config)
    init_profiler(config, aim_logger, rank)

    train_loader, val_loader = get_dataloaders(config, aim_logger, rank)

    net, opt, lr_scheduler, ema_model, resume_epoch, scaler = build_model(config, rank)

    trainer = Trainer(
        device=device,
        max_epochs=config.training.epochs,
        data_loader=train_loader,
        prepare_batch=prepare_batch,
        iteration_update=train_step,
        # key_metric={'loss': None},           # or your dict of Metrics
        additional_metrics=None,
        metric_cmp_fn=default_metric_cmp_fn,
        amp=config.amp.enabled,
    )
    trainer.network = net
    trainer.optimizer = opt
    trainer.lr_scheduler = lr_scheduler
    trainer.scaler_ = scaler
    trainer.config = config

    trainer.ce = nn.CrossEntropyLoss()
    trainer.mse = nn.MSELoss()
    trainer.bce = nn.BCEWithLogitsLoss()
    trainer.dice_loss = DiceLoss(sigmoid=True)
    # trainer.dice_loss = DiceLoss(softmax=True, to_onehot_y=True)

    trainer.optim_set_to_none = config.optimizer.set_to_none

    # ________Validation Setup________

    postprocess = transforms.Compose(
        [
            transforms.AsDiscreted(
                keys=Keys.PRED,
                argmax=True,
                to_onehot=config.data.num_classes,
                # keys=Keys.PRED, threshold=0.5
            ),
        ]
    )
    metrics = {
        "Mean Dice": MeanDice(
            include_background=False,
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=config.data.num_classes,
        ),
        # "Dummy Metric": lambda x: (logging.info(f"Dummy metric called with {x}"), 0.0)[1],
    }
    val_evaluator = Evaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=prepare_batch,
        iteration_update=eval_step,
        postprocessing=postprocess,
        key_val_metric=metrics,
    )
    # 4) Attach objects needed during eval
    val_evaluator.network = ema_model
    val_evaluator.config = config
    val_evaluator.inferer = SlidingWindowInferer(
        roi_size=config.data.roi_size, sw_batch_size=1, overlap=0.5
    )

    attach_handlers(
        trainer,
        val_evaluator,
        net,
        opt,
        lr_scheduler,
        scaler,
        ema_model,
        config,
        aim_logger,
        val_loader,
        rank,
        postprocess=postprocess,
    )
    # Add barrier to ensure all processes are ready before starting training
    idist.utils.barrier()

    if rank == 0:
        logging.info(
            f"[Rank {rank}] Starting training for {config.training.epochs} epochs"
        )
        logging.info(
            f"[Rank {rank}] Training on {len(train_loader.dataset)} training samples"
        )
        logging.info(
            f"[Rank {rank}] Validation on {len(val_loader.dataset)} validation samples"
        )

    
    with TorchProfiler(subdir="trainer_run"):
        trainer.state.epoch = resume_epoch
        if resume_epoch > 0:
            logging.info(f"[Rank {rank}] Resuming training from epoch {resume_epoch}")
       
        # val_evaluator.run() # Run validation before starting training
        trainer.run()


if __name__ == "__main__":
    config = get_args()
    # world_size = torch.cuda.device_count()
    # rank = int(os.environ["LOCAL_RANK"])  # torchrun provides this env var
    # logging.info(f"Running on rank {rank} with world size {world_size}")
    # main(rank, world_size)

    if DEBUG:
        config.experiment.name = f"debug_{config.experiment.name}"
        config.training.save_dir = os.path.join(
            config.training.save_dir, "debug"
        )
        config.evaluation.validation_interval = 1
        config.experiment.tags.append("debug")
        config.training.resume = None  # Don't resume from any previous run

        logging.info(
            f""" {'-'* 50}
                     DEBUG MODE:
                    - Using only 10 training and validation samples.
                    - Validation interval set to 1 epoch.
                    {'-'* 50}
                     """
        )

    with idist.Parallel(
        backend="nccl", nproc_per_node=torch.cuda.device_count()
    ) as parallel:
        parallel.run(_distributed_run, config)
