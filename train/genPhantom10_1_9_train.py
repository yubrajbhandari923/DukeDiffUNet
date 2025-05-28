import logging
import sys, os
from glob import glob
import time

import monai.data
import monai.transforms
import numpy as np
import torch
from ignite.engine import (
    Events,
    _prepare_batch,
)
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader

import SimpleITK as sitk

import monai
from monai.data import list_data_collate, decollate_batch
from monai.handlers import (
    MeanDice,
    StatsHandler,
    IgniteMetricHandler,
    stopping_fn_from_metric,
    from_engine,
)
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    CropForegroundd,
    SaveImage,
    Orientationd,
    ResizeWithPadOrCropd,
    CropForegroundd,
    Flipd,
    ToDeviced,
    ToTensord,
)
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.utils.enums import CommonKeys as Keys

from generative_phantom.monai_transforms import (
    CombineStructuresd,
    CropBelowIndexd,
    CropRandomlyBetweenIndexd,
    CropAboveIndexd,
)
from generative_phantom.model import EncoderDecoderModel, DiceLoss
from generative_phantom.data_preparation import dukesegv2_mapping_inv
from generative_phantom.monai_helpers import AimIgniteImageHandler

from generative_phantom.configs.model_config import (
    ModelConfig,
    PreprocessedDataConfig,
    transforms_to_str,
)

from aim.pytorch_ignite import AimLogger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

CODE_TESTING = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(
        name="genPhantom10_1_9",
        description="Same as 10.1.7 but with vertebrae_c1 and vertebrae_c2 retained",
        save_path="/scratch/railabs/yb107/genPhantom/results/genPhantom10.1.9",
        preprocessed_config_path="/scratch/railabs/yb107/genPhantom/preprocessed/skull_training_data/config.json",
        epochs=1500,
        batch_size=1,
        num_workers=12,
        device="cuda",
        output_channel=5 + 1,  # Fix
        loss="DiceLoss",
        optimizer="Adam",
        patch_size=(256, 256, 256),
        patch_overlap=0,
        lr=0.001,
        beta_1=0.3,
        beta_2=0.999,
        early_stopping=True,
        early_stopping_patience=70,
        log_interval=1,
        validation_interval=2,
        visualization_interval=1,
        # log_dir="/scratch/railabs/yb107/genPhantom/results",
        aim_repo_dir="/scratch/railabs/yb107/genPhantom/results/aim",
        additional_params={
            "genPhantom10_3_map": {
                "background": 0,
                "skull": 1,
                "brain": 2,
                "vertebrae_c1": 3,
                "vertebrae_c2": 4,
                # -----------------
            }
        },
        model="EncoderDecoderModel sum_res=False",
    )
    monai.config.print_config()

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # Data preparation
    train_data_config = PreprocessedDataConfig.from_json(
        config.preprocessed_config_path
    )
    data_images = glob(train_data_config.save_dir + "/*.nii.gz")
    data_images = [
        i
        for i in data_images
        if os.path.basename(i).split("_trans.nii.gz")[0]
        in train_data_config.additional_params["cases_with_full_skulls"]
    ]

    if CODE_TESTING:
        data_images = data_images[:10]

    train_images = data_images[: int(len(data_images) * 0.8)]
    val_images = data_images[int(len(data_images) * 0.8) :]

    train_files = [
        {Keys.IMAGE: img, Keys.LABEL: seg}
        for img, seg in zip(train_images, train_images)
    ]

    val_files = [
        {Keys.IMAGE: img, Keys.LABEL: seg} for img, seg in zip(val_images, val_images)
    ]

    genPhantom10_3_map = config.additional_params["genPhantom10_3_map"]
    genPhantom10_3_mapping = lambda x: genPhantom10_3_map.get(x.lower(), 5)

    train_transforms = Compose(
        [
            LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Orientationd(keys=[Keys.IMAGE, Keys.LABEL], axcodes="LPI"),
            CropForegroundd(keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.LABEL),
            CropRandomlyBetweenIndexd(
                keys=[Keys.IMAGE], top_index=1, bottom_index=27
            ),  # Erase normally above vertebrae_c7
            CropBelowIndexd([Keys.IMAGE, Keys.LABEL], 24),  # Crop below vertebrae_t5
            CombineStructuresd(
                [Keys.IMAGE, Keys.LABEL], dukesegv2_mapping_inv, genPhantom10_3_mapping
            ),
            # ToDeviced([Keys.IMAGE, Keys.LABEL], device=device),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Flipd(keys=[Keys.IMAGE, Keys.LABEL], spatial_axis=-1),
            ResizeWithPadOrCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.patch_size,
                method="end",
            ),
            Flipd(keys=[Keys.IMAGE, Keys.LABEL], spatial_axis=-1),
            ToTensord(
                keys=[Keys.IMAGE, Keys.LABEL],
                dtype=torch.float32,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Orientationd(keys=[Keys.IMAGE, Keys.LABEL], axcodes="LPI"),
            CropForegroundd(keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.LABEL),
            CropRandomlyBetweenIndexd(
                keys=[Keys.IMAGE], top_index=1, bottom_index=27, seed="filename"
            ),  # Erase normally above vertebrae_c7
            CropBelowIndexd([Keys.IMAGE, Keys.LABEL], 24),  # Crop below vertebrae_t5
            CombineStructuresd(
                [Keys.IMAGE, Keys.LABEL], dukesegv2_mapping_inv, genPhantom10_3_mapping
            ),
            # ToDeviced([Keys.IMAGE, Keys.LABEL], device=device),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            Flipd(keys=[Keys.IMAGE, Keys.LABEL], spatial_axis=-1),
            ResizeWithPadOrCropd(
                keys=[Keys.IMAGE, Keys.LABEL],
                spatial_size=config.patch_size,
                method="end",
            ),
            Flipd(keys=[Keys.IMAGE, Keys.LABEL], spatial_axis=-1),
            ToTensord(
                keys=[Keys.IMAGE, Keys.LABEL],
                dtype=torch.float32,
            ),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # create UNet, DiceLoss and Adam optimizer
    net = EncoderDecoderModel(output_chn=config.output_channel, sum_res=False).to(
        device
    )
    loss = DiceLoss(output_chn=config.output_channel, scaling_factor=2e7)
    opt = torch.optim.Adam(net.parameters(), config.lr)

    config.model = net.__class__.__name__
    config.pytorch_version = str(torch.__version__)
    config.monai_version = str(monai.__version__)
    config.train_transforms = transforms_to_str(train_transforms)
    config.train_length = len(train_ds)
    config.val_length = len(val_ds)

    if not CODE_TESTING:
        config.save(config.save_path)

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=True):
        return _prepare_batch(
            (batch[Keys.IMAGE], batch[Keys.LABEL]), device, non_blocking
        )

    # This part is ignite-specific
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=config.epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        prepare_batch=prepare_batch,
    )

    metrics = {
        "Mean Dice": MeanDice(
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=config.output_channel,
        ),
        "Volume Generated": IgniteMetricHandler(
            # loss_fn be a function that takes y_pred and returns batch-first tensor with volume of each class
            loss_fn=lambda y_pred, y: torch.sum(y_pred > 0, dim=(-1, -2, -3))
            / 1000,  # volume in ml
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
        ),
    }

    post_pred = Compose(
        [
            AsDiscreted(keys=Keys.LABEL, to_onehot=config.output_channel),
            AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=config.output_channel),
        ]
    )
    val_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        prepare_batch=prepare_batch,
        key_val_metric=metrics,
        postprocessing=post_pred,
        non_blocking=True,
    )

    if not CODE_TESTING:
        # adding checkpoint handler to save models (network params and optimizer stats) during training
        checkpoint_handler = ModelCheckpoint(
            config.save_path + "/checkpoints",
            filename_prefix=config.name,
            n_saved=None,
            require_empty=False,
            global_step_transform=lambda eng, event: eng.state.epoch,
        )
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=checkpoint_handler,
            to_save={"network": net, "optimizer": opt},
        )

    train_stats_handler = StatsHandler(
        name="trainer",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
    )
    train_stats_handler.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED(every=config.validation_interval))
    def run_validation(engine):
        val_evaluator.run()

    # add early stopping handler to evaluator
    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        score_function=stopping_fn_from_metric("Mean Dice"),
        trainer=trainer,
    )

    val_evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=early_stopper
    )

    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
        iteration_log=False,
    )
    val_stats_handler.attach(val_evaluator)

    # Create a logger
    if CODE_TESTING:
        aim_logger = AimLogger(
            repo=config.aim_repo_dir,
            experiment="Code Testing",
        )
        aim_logger.experiment.add_tag("code_testing")
    else:
        aim_logger = AimLogger(
            repo=config.aim_repo_dir,
            experiment=config.name,
        )
        aim_logger.experiment.add_tag("Training")
        aim_logger.experiment.description = config.description

    aim_logger.log_params(config.__dict__)
    aim_logger.experiment.log_warning(
        "WARNING: Number of Val images is not divisible by 4"
        if len(val_loader) % 4 != 0
        else "CHECKED: Number of Val images is divisible by 4"
    )

    # aim_logger.attach(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED,
    #     log_handler=AimIgniteImageHandler(
    #         "Input",
    #         output_transform=from_engine([Keys.IMAGE], first=True),
    #         global_step_transform=global_step_from_engine(trainer),
    #     )
    # )

    # aim_logger.attach(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED,
    #     log_handler=AimIgniteImageHandler(
    #         "Label",
    #         output_transform=from_engine([Keys.LABEL], first=True),
    #         global_step_transform=global_step_from_engine(trainer),
    #     )
    # )

    aim_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="Iteration Dice Loss",
        output_transform=from_engine(["loss"], first=True),
    )
    aim_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="Epoch Dice Loss",
        output_transform=from_engine(["loss"], first=True),
    )

    aim_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=["Mean Dice", "Volume Generated"],
        global_step_transform=global_step_from_engine(trainer),
    )

    aim_logger.attach(
        val_evaluator,
        AimIgniteImageHandler(
            "Prediction",
            output_transform=from_engine([Keys.PRED], first=True),
            global_step_transform=global_step_from_engine(trainer),
        ),
        event_name=Events.ITERATION_COMPLETED(
            every=2 if (len(val_loader) // 4) == 0 else len(val_loader) // 4
        ),
    )

    aim_logger.attach(
        val_evaluator,
        AimIgniteImageHandler(
            "Label",
            output_transform=from_engine([Keys.LABEL], first=True),
            global_step_transform=global_step_from_engine(trainer),
            plot_once=True,
            log_unique_values=False,
        ),
        event_name=Events.ITERATION_COMPLETED(
            every=2 if (len(val_loader) // 4) == 0 else len(val_loader) // 4
        ),
    )

    aim_logger.attach(
        val_evaluator,
        AimIgniteImageHandler(
            "Image",
            output_transform=from_engine([Keys.IMAGE], first=True),
            global_step_transform=global_step_from_engine(trainer),
            plot_once=True,
            log_unique_values=False,
        ),
        event_name=Events.ITERATION_COMPLETED(
            every=2 if (len(val_loader) // 4) == 0 else len(val_loader) // 4
        ),
    )

    # vol_info = lambda x: f"Image: {torch.sum(x[Keys.IMAGE] > 0) / 1000}; Label: {torch.sum(x[Keys.LABEL] > 0) / 1000}"
    # aim_logger.experiment.log_info("Validation Volume Info: \n"+ "\n".join([vol_info(x) for x in val_loader]))
    # aim_logger.experiment.log_info("Training Volume Info: \n"+ "\n".join([vol_info(x) for x in train_loader]))
    aim_logger.experiment.log_info(
        f"{len(train_loader)} training batches, {len(val_loader)} validation batches"
    )

    # if not CODE_TESTING:
    #     # Save 10 images for visualization
    #     logging.info("Train Image Saving")

    #     for idx, img in enumerate(train_loader):
    #         logging.info(f"Image: {img[Keys.IMAGE].shape}; Label: {img[Keys.LABEL].shape}")

    #         assert img[Keys.IMAGE].shape == img[Keys.LABEL].shape
    #         # assert if overlap between image and label is non zero
    #         assert torch.sum((img[Keys.IMAGE] > 0) == (img[Keys.LABEL] > 0)) > 0

    #         SaveImage( output_dir=f"{config.save_path}/input/train", output_postfix="input", resample=False, separate_folder=False)(img[Keys.IMAGE][0])
    #         SaveImage( output_dir=f"{config.save_path}/input/train", output_postfix="label", resample=False, separate_folder=False)(img[Keys.LABEL][0])
    #         if idx == 5:
    #             break

    #     logging.info("Validation Images Saving")
    #     for idx, img in enumerate(val_loader):
    #         logging.info(f"Image: {img[Keys.IMAGE].shape}; Label: {img[Keys.LABEL].shape}")

    #         assert img[Keys.IMAGE].shape == img[Keys.LABEL].shape
    #         assert torch.sum((img[Keys.IMAGE] > 0) == (img[Keys.LABEL] > 0)) > 0

    #         SaveImage( output_dir=f"{config.save_path}/input/val", output_postfix="input", resample=False, separate_folder=False)(img[Keys.IMAGE][0])
    #         SaveImage( output_dir=f"{config.save_path}/input/val", output_postfix="label", resample=False, separate_folder=False)(img[Keys.LABEL][0])
    #         if idx == 5:
    #             break

    trainer.run()


if __name__ == "__main__":
    main()
