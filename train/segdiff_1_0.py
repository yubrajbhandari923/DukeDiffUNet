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
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
)
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import SimpleInferer

from monai.handlers import (
    MeanDice,
    StatsHandler,
    IgniteMetricHandler,
    stopping_fn_from_metric,
    from_engine,
)
from monai.data import list_data_collate, decollate_batch, NumpyReader
from monai.data import GridPatchDataset, PatchIterd
from monai.losses import DiceLoss
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines import Evaluator
from monai.utils.enums import CommonKeys as Keys

from utils.monai_helpers import AimIgniteImageHandler
from utils.monai_transforms import DropKeysd
from model.segdiff import SegDiffModel
from model.segdiff.sampling_util import sampling_major_vote_func

from utils.profiling import profile_block, TorchProfiler, init_profiler
from torch.utils.data.distributed import DistributedSampler

from aim.pytorch_ignite import AimLogger
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import functools

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def ddp_setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


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


class DiffusionTrainer(SupervisedTrainer):
    def __init__(
        self,
        *,
        diffusion,
        schedule_sampler,
        **super_kwargs,  # everything SupervisedTrainer normally gets
    ):
        super().__init__(**super_kwargs)
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler

    @profile_block("train_iteration")
    def _iteration(self, engine, batchdata):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        # 1) prepare inputs & targets exactly as MONAI does
        batch = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs
        )
        if len(batch) == 2:
            inputs, targets = batch
            args, kwargs = (), {}
        else:
            inputs, targets, args, kwargs = batch

        inputs = inputs.to(engine.state.device, non_blocking=engine.non_blocking)
        targets = targets.to(engine.state.device, non_blocking=engine.non_blocking)

        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        # in the original segDiff TrainLoop they have x_t is from segmentation mask and {"conditioned_image": Image}

        # 2) sample timesteps & weights
        t, weights = self.schedule_sampler.sample(targets.shape[0], engine.state.device)

        # 3) compute the diffusion losses
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            engine.network,
            targets,
            t,
            model_kwargs={
                "conditioned_image": inputs
            },  # or fill in conditioning if you have labels
        )
        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()

        # store pred if your diffusion returns it
        engine.state.output[Keys.PRED] = losses.get("pred", torch.zeros_like(inputs))
        engine.state.output[Keys.LOSS] = loss

        # 4) standard backward + step
        engine.network.train()
        engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        loss.backward()
        engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
        engine.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output


class DiffusionEvaluator(Evaluator):
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
        mode="eval",
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

    @profile_block("eval_iteration")
    def _iteration(self, engine, batchdata):
        # 1) prepare image and ground truth mask
        inputs, targets = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking
        )
        # engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}
        engine.state.output = {'y': targets}

        # 2) perform major-vote sampling
        with torch.no_grad():
            # accumulate predictions
            votes = []
            for _ in range(self.n_rounds):
                # sample timesteps for each round
                # p_sample_loop will loop internally over timesteps
                x = self.diffusion.p_sample_loop(
                    self.network,
                    (self.major_vote, targets.shape[1], *inputs.shape[2:]),
                    noise=None,
                    clip_denoised=False,
                    denoised_fn=None,
                    model_kwargs={"conditioned_image": inputs},
                    device=engine.state.device,
                    progress=False,
                )
                # normalize from [-1,1] to [0,1]
                x = (x + 1.0) / 2.0
                # average across vote dimension and round to binary
                vote_mean = x.mean(dim=0, keepdim=True).round()
                votes.append(vote_mean)
            # final vote: mean of all rounds
            final_vote = torch.stack(votes).mean(dim=0).round()

        # 3) store prediction
        # engine.state.output[Keys.PRED] = final_vote
        engine.state.output['y_pred'] = final_vote

        # 4) save as NIfTI
        # wrap in MetaTensor to preserve spatial metadata
        # SaveImage will read metadata from inputs
        engine.state.output[Keys.PRED] = final_vote.cpu()

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


@profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger, rank, world_size):
    train_files = []
    val_files = []

    # load the training and validation data
    with open(config.data.train_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            train_files.append({Keys.IMAGE: data["data"]})
    train_files = train_files[:5]
    
    if aim_logger is not None and rank == 0:
        aim_logger.experiment.log_info(
            f"Training files {json.dumps(train_files, indent=2)}"
        )
        logging.info(f"Training files length: {len(train_files)}")
    
        
    with open(config.data.val_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            val_files.append({Keys.IMAGE: data["data"]})
    
    if aim_logger is not None and rank == 0:    
        aim_logger.experiment.log_info(
            f"Validation files {json.dumps(val_files, indent=2)}"
        )
        logging.info(f"Validation files length: {len(val_files)}")

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
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            RandSpatialCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                roi_size=config.data.roi_size,
                random_size=False,
            ),
            ToTensord(keys=[Keys.IMAGE, Keys.LABEL]),
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
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL],
                pixdim=config.data.pixdim,
                mode=["bilinear", "nearest"],
            ),
            # RandSpatialCropSamplesd(
            #     keys=[Keys.IMAGE, Keys.LABEL],
            #     roi_size=config.data.roi_size,
            #     random_size=False,
            #     num_samples=config.data.val_num_samples,
            # )
            RandCropByPosNegLabeld(
                keys=[Keys.IMAGE, Keys.LABEL],
                label_key=Keys.LABEL,
                spatial_size=config.data.roi_size,
                pos=4,
                neg=1,
            ),
            # lambda data: (del data["Spacing"], data)[1],  # remove spacing key
            # DropKeysd(keys=["Spacing"]),
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
    # val_ds = GridPatchDataset(
    #     data=val_files,
    #     patch_iter=PatchIterd(
    #         keys=[Keys.IMAGE, Keys.LABEL],
    #         patch_size=config.data.roi_size,
    #         mode="constant",
    #     ),
    #     transform=val_transforms,
    # )
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
    segdiff = SegDiffModel(OmegaConf.resolve(config.model))

    net = segdiff.get_model().to(rank)
    net = DDP(net, device_ids=[rank])

    diffusion = segdiff.get_diffusion()
    schedule_sampler = segdiff.get_schedule_sampler(config.model.schedule_sampler)

    loss_fn = DiffusionLossWrapper(net, diffusion, schedule_sampler)

    opt = optim.AdamW(
        net.parameters(),
        config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )
    return net, loss_fn, opt, diffusion, schedule_sampler


def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    # cond = {}  # Customize if using labels or other inputs as conditions
    labels = batch.get(Keys.LABEL)
    return images, labels

def prepare_val_batch(batch, device=None, non_blocking=True):
    images = batch[0][Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[0].get(Keys.LABEL)
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


def attach_checkpoint_handler(trainer, net, opt, config, rank):
    if rank != 0:
        return
    
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
    def update_(engine):
        for p_ema, p in zip(ema_params[0], net.parameters()):
            update_ema(p_ema, p, rate=config.model.ema_rate)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=10), update_)


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
    if rank != 0:
        return
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


def main(rank, world_size):
    ddp_setup(rank, world_size)
    config = get_args()

    config.train.device = f"cuda:{rank}"

    init_profiler(config)

    log_config(config, rank)
    torch.random.manual_seed(config.train.seed)

    if rank == 0:
        aim_logger = get_aim_logger(config)
    else:
        aim_logger = None

    train_loader, val_loader = get_dataloaders(config, aim_logger, rank, world_size)

    net, loss, opt, diffusion, schduler = build_model(config, rank)

    ema_params = [copy.deepcopy(list(net.parameters())) for _ in range(1)]

    # trainer = SupervisedTrainer(
    #     device=config.train.device,
    #     max_epochs=config.train.epochs,
    #     train_data_loader=train_loader,
    #     network=net,
    #     optimizer=opt,
    #     loss_function=loss,
    #     prepare_batch=prepare_batch,
    # )
    trainer = DiffusionTrainer(
        device=config.train.device,
        max_epochs=config.train.epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        prepare_batch=prepare_batch,
        diffusion=diffusion,
        schedule_sampler=schduler,
    )

    # val_evaluator = SupervisedEvaluator(
    #     device=config.train.device,
    #     val_data_loader=val_loader,
    #     network=net,
    #     inferer=monai.inferers.SlidingWindowInferer(
    #         roi_size=config.data.roi_size,
    #         sw_batch_size=config.data.batch_size,
    #         overlap=0.5,
    #     ),
    #     prepare_batch=prepare_batch,
    #     key_val_metric=get_metrics(),
    #     postprocessing=get_post_processing(),
    #     non_blocking=True,
    # )

    post_pred = AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=2)
    metrics = {"Mean Dice": MeanDice(include_background=False)}

    val_evaluator = DiffusionEvaluator(
        device=config.train.device,
        val_data_loader=val_loader,
        diffusion=diffusion,
        network=net,
        prepare_batch=default_prepare_batch,
        schedule_sampler=schduler,
        postprocessing=post_pred,
        key_val_metric=metrics,
    )

    attach_checkpoint_handler(trainer, net, opt, config)

    attach_stats_handlers(trainer, val_evaluator, config)

    attach_ema_update(trainer, net, ema_params, config)

    attach_ema_validation(trainer, net, ema_params, val_evaluator, val_loader, config)

    # attach_validation(trainer, val_evaluator, config)

    attach_aim_handlers(trainer, val_evaluator, aim_logger, val_loader)

    # val_evaluator.run()
    with TorchProfiler(config, subdir="trainer_run"):
        trainer.run()

    cleanup()


if __name__ == "__main__":
    main()
