import os
import sys
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import json
import copy
import numpy as np
import argparse


import monai
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Spacingd,
    EnsureChannelFirstd,
    AsDiscreted,
    ResizeWithPadOrCropd,
)
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader

from monai.handlers import (
    MeanDice,
    StatsHandler,
    IgniteMetricHandler,
    stopping_fn_from_metric,
    from_engine,
)
from monai.data import list_data_collate, decollate_batch, NumpyReader
from monai.losses import DiceLoss
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.utils.enums import CommonKeys as Keys

from utils.monai_helpers import AimIgniteImageHandler
from model.segdiff import SegDiffModel

from aim.pytorch_ignite import AimLogger
from omegaconf import OmegaConf

import functools

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# region utils (Diffusion loss wrapper, EMA update)


class DiffusionLossWrapper:
    def __init__(self, model, diffusion, schedule_sampler):
        self.model = model
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler

    def __call__(self, images, cond=None):
        t, weights = self.schedule_sampler.sample(images.shape[0], device=images.device)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            images,
            t,
            model_kwargs=cond,
        )

        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()
        return loss


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
        default="/home/yb107/cvit-work/cvpr2025/DukeDiffSeg/configs/base.yaml",
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


# region Logging and Config Handling (log_config, get_aim_logger, get_dataloaders, build_model, prepare_batch, get_metrics, get_post_processing)


def log_config(config):
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

    # Log this file's code to the aim_logger # Fix this
    # aim_logger.experiment.log_code(
    #     os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)), "experiment_1.py"
    #     ),
    #     name="experiment_1.py",
    # )

    # config.aim_logger = aim_logger

    return aim_logger


def get_dataloaders(config, aim_logger):
    train_files = []
    val_files = []

    # load the training and validation data
    with open(config.data.train_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            train_files.append({Keys.IMAGE: data["data"]})

    aim_logger.experiment.log_info(
        f"Training files {json.dumps(train_files, indent=2)}"
    )

    with open(config.data.val_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            val_files.append({Keys.IMAGE: data["data"]})
    aim_logger.experiment.log_info(
        f"Validation files {json.dumps(val_files, indent=2)}"
    )

    def set_spacing(meta_tensor, spacing):
        spacing = (
            spacing.tolist()
            if isinstance(spacing, (np.ndarray, torch.Tensor))
            else spacing
        )
        meta_tensor.meta["spacing"] = spacing
        return meta_tensor

    train_transforms = Compose(
        [
            lambda x: {
                Keys.IMAGE: x[Keys.IMAGE],
                Keys.LABEL: x.get(Keys.IMAGE),
                "spacing": x.get(Keys.IMAGE),
            },
            LoadImaged(keys=[Keys.IMAGE], reader=NumpyReader, npz_keys=["imgs"]),
            LoadImaged(keys=[Keys.LABEL], reader=NumpyReader, npz_keys=["gts"]),
            lambda data: {
                Keys.IMAGE: data[Keys.IMAGE],
                Keys.LABEL: data.get(Keys.LABEL),
                "Spacing": np.load(data["spacing"])["spacing"],
            },
            lambda x: {
                Keys.IMAGE: set_spacing(x[Keys.IMAGE], x["Spacing"]),
                Keys.LABEL: set_spacing(x[Keys.LABEL], x["Spacing"]),
            },
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            RandSpatialCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                roi_size=config.data.roi_size,
                random_size=False,
            ),
        ]
    )
    val_transforms = Compose(
        [
            lambda x: {
                Keys.IMAGE: x[Keys.IMAGE],
                Keys.LABEL: x.get(Keys.IMAGE),
                "spacing": x.get(Keys.IMAGE),
            },
            LoadImaged(keys=[Keys.IMAGE], reader=NumpyReader, npz_keys=["imgs"]),
            LoadImaged(keys=[Keys.LABEL], reader=NumpyReader, npz_keys=["gts"]),
            lambda data: {
                Keys.IMAGE: data[Keys.IMAGE],
                Keys.LABEL: data.get(Keys.LABEL),
                "Spacing": np.load(data["spacing"])["spacing"],
            },
            lambda x: {
                Keys.IMAGE: set_spacing(x[Keys.IMAGE], x["Spacing"]),
                Keys.LABEL: set_spacing(x[Keys.LABEL], x["Spacing"]),
            },
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model(config):
    segdiff = SegDiffModel(OmegaConf.resolve(config.model))

    net = segdiff.get_model()
    diffusion = segdiff.get_diffusion()
    schedule_sampler = segdiff.get_schedule_sampler(config.model.schedule_sampler)

    loss_fn = DiffusionLossWrapper(net, diffusion, schedule_sampler)

    opt = optim.AdamW(
        net.parameters(),
        config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )
    return net, loss_fn, opt


def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    cond = {}  # Customize if using labels or other inputs as conditions
    return images, cond


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


def attach_checkpoint_handler(trainer, net, opt, config):
    checkpoint_handler = ModelCheckpoint(
        dirname=os.path.join(config.train.save_dir, "checkpoints"),
        filename_prefix=config.name,
        n_saved=None,
        require_empty=False,
        global_step_transform=lambda eng, _: eng.state.epoch,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {"network": net, "optimizer": opt}
    )


def attach_ema_update(trainer, net, ema_params, config):
    def update_ema(engine):
        for p_ema, p in zip(ema_params[0], net.parameters()):
            update_ema(p_ema, p, rate=config.model.args.ema_rate)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=10), update_ema)


def attach_validation(trainer, val_evaluator, config):
    def run_validation(engine):
        val_evaluator.run()

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.train.eval.validation_interval),
        run_validation,
    )


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


def attach_stats_handlers(trainer, val_evaluator, config):
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


def attach_early_stopping(val_evaluator, trainer, config):
    stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        score_function=stopping_fn_from_metric("Mean Dice"),
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopper)


def attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader):
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


# endregion


def main():
    config = get_args()

    log_config(config)
    torch.random.manual_seed(config.train.seed)

    aim_logger = get_aim_logger(config)

    train_loader, val_loader = get_dataloaders(config, aim_logger)

    net, loss, opt = build_model(config)

    ema_params = [copy.deepcopy(list(net.parameters())) for _ in range(1)]

    trainer = SupervisedTrainer(
        device=config.train.device,
        max_epochs=config.train.epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        prepare_batch=prepare_batch,
    )

    val_evaluator = SupervisedEvaluator(
        device=config.train.device,
        val_data_loader=val_loader,
        network=net,
        inferer=monai.inferers.SlidingWindowInferer(
            roi_size=config.data.roi_size,
            sw_batch_size=config.data.batch_size,
            overlap=0.5,
        ),
        prepare_batch=prepare_batch,
        key_val_metric=get_metrics(),
        postprocessing=get_post_processing(),
        non_blocking=True,
    )

    attach_checkpoint_handler(trainer, net, opt, config)

    attach_stats_handlers(trainer, val_evaluator, config)

    attach_ema_update(trainer, net, ema_params, config)

    attach_ema_validation(trainer, net, ema_params, val_evaluator, val_loader, config)

    attach_validation(trainer, val_evaluator, config)

    attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader)

    trainer.run()


if __name__ == "__main__":
    main()
