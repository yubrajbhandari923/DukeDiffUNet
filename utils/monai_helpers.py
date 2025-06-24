from typing import Any
from monai.metrics.metric import Metric, CumulativeIterationMetric
import logging
import torch
import json

# from ignite.handlers.base_logger import BaseHandler
from aim import Figure, Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ignite.engine.events import Events

from aim.sdk.sequence_collection import SingleRunSequenceCollection
from aim.sdk.sequences.figure_sequence import Figures

import SimpleITK as sitk
from pathlib import Path

import multiprocessing
import os
import numpy as np

import matplotlib.pyplot as plt


class DiceScoreMetric(Metric):
    def __init__(self, device=None, smooth=1e-5):
        # self.output_transform = output_transform
        self.device = device
        self.smooth = smooth

    def __call__(self, y_pred, y):
        # y_pred, y = self.output_transform(output)

        logging.info(f"y_pred: {y_pred.shape}, y: {y.shape}")
        output_chn = y_pred.shape[1]

        smooth = self.smooth
        dice = 0
        for i in range(output_chn):
            y_pred_tensor = torch.empty_like(y_pred[:, i, :, :, :], device=self.device)
            # y_tensor = torch.empty_like(y[:, i, :, :, :], device=self.device)

            y_pred_tensor.copy_(y_pred[:, i, :, :, :])
            # y_tensor.copy_(y[:, i, :, :, :])
            y_tensor = y[:, i, :, :, :]

            # y_pred_tensor = torch.tensor(y_pred[:, i, :, :, :], device=self.device)
            # y_tensor = torch.tensor(y[:, i, :, :, :], device=self.device)

            inse = (y_pred_tensor * y_tensor).mean()
            l = (y_pred_tensor * y_pred_tensor).sum()
            r = (y_tensor * y_tensor).sum()
            # inse = (y_pred[:, i, :, :, :] * y[:, i, :, :, :]).mean()
            # l = (y_pred[:, i, :, :, :]*y_pred[:,i,:,:,:]).sum()
            # r = (y[:, i, :, :, :]*y[:,i,:,:,:]).sum()
            if l + r == 0:
                logging.info(f"L+r: 0, inse: {inse}, output_channel: {i}")
            dice += 2.0 * (inse + smooth) / (l + r + smooth)

        dice = (dice / output_chn).detach()
        logging.info(f"Dice: {dice}")
        return dice


class AimIgnite3DImageHandler:
    """
    Ignite Image Handler for AIM.

    """

    plotted_tags = set()
    last_printed_unique_values = torch.tensor([])

    def __init__(
        self,
        tag,
        output_transform=None,
        global_step_transform=None,
        plot_once=False,
        log_unique_values=True,
    ):
        self.tag = tag
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform
        self.plot_once = plot_once
        self.log_unique_values = log_unique_values

    def __call__(self, engine, logger, event_name):

        if self.output_transform is not None:
            img = self.output_transform(engine.state.output)

        img_name = (
            img.meta["filename_or_obj"]
            .split("/")[-1]
            .split(".")[0]
            .replace("_trans", "")
        )
        tag_name = f"{self.tag} {' '.join(img_name.split('_')[:2])}"

        if self.plot_once and tag_name in AimIgnite3DImageHandler.plotted_tags:
            return

        if len(img.shape) == 5 or len(img.shape) == 4:
            img = img.squeeze()

        if len(img.shape) == 4:
            img = torch.argmax(img, dim=0)

        img_data = img.cpu().numpy()

        volume = np.sum(img_data[img_data > 0]) / 1000
        unique_values = torch.unique(img).cpu()

        fig = get_big_fig(
            img_data,
            title=img_name,
            table_data={
                "header": ["Key", "Value"],
                "data": [
                    [
                        "Tag",
                        "Shape",
                        "Total Volume",
                        "Unique Values",
                        "Full Patient Name",
                    ],
                    [self.tag, str(img_data.shape), volume, unique_values, img_name],
                ],
            },
            n_every=5,
        )

        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, Events.EPOCH_COMPLETED)
        else:
            global_step = engine.state.get_event_attrib_value(event_name)

        logger.experiment.track(Figure(fig), name=tag_name, step=global_step)

        if self.log_unique_values:
            unique_values = torch.unique(img).cpu()
            if not AimIgnite3DImageHandler.last_printed_unique_values.equal(
                unique_values
            ):
                logger.experiment.log_info(
                    f"""Epoch: {global_step},
                    Unique values for {self.tag}: {unique_values}"""
                )

        AimIgnite3DImageHandler.plotted_tags.add(tag_name)


class AimIgnite2DImageHandler:
    """
    Ignite Image Handler for AIM.

    """

    plotted_tags = set()

    def __init__(
        self,
        tag,
        output_transform=None,
        global_step_transform=None,
        plot_once=False,
    ):
        self.tag = tag
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform
        self.plot_once = plot_once

    def __call__(self, engine, logger, event_name):

        if self.output_transform is not None:
            data = self.output_transform(engine.state.output)
            # ([Images], [Labels], [Predictions])

        images = data[0]  # Assuming data is a tuple of (images, labels, predictions)
        label = data[1]
        pred = data[2]

        for  img, lbl, prd in zip(images, label, pred):
            img_name = (
                img.meta["filename_or_obj"]
                .split("/")[-1]
                .split(".")[0]
                .replace("_trans", "")
            )
            tag_name = f"{self.tag} {' '.join(img_name.split('_')[:2])}"

            if self.plot_once and tag_name in AimIgnite2DImageHandler.plotted_tags:
                return

            # if len(img.shape) == 5 or len(img.shape) == 4:
            img = img.squeeze()
            lbl = lbl.squeeze()
            prd = prd.squeeze()

            if len(img.shape) > 2:
                raise ValueError(
                    f"Image shape {img.shape}, Label Shape {lbl.shape}, Pred Shape {prd.shape} is not 2D. Expected 2D images for visualizations."
                )
                
            logging.info(f"Unique values in label: {torch.unique(lbl)}, Unique values in pred: {torch.unique(prd)}")

            # fig = plot_2d_images(img_name, img_data, label_data, pred_data)
            fig = plot_image_label_pred(
                img_name, img, lbl, prd
            )
            if self.global_step_transform is not None:
                global_step = self.global_step_transform(engine, Events.EPOCH_COMPLETED)
            else:
                global_step = engine.state.get_event_attrib_value(event_name)
            logger.experiment.track(Image(fig), name=tag_name, step=global_step)
            AimIgnite2DImageHandler.plotted_tags.add(tag_name)


def normalize_uint8(x):
    x = np.nan_to_num(x)  # clean NaNs
    x = x - np.min(x)
    x = x / (np.max(x) + 1e-8)
    return (x * 255).astype(np.uint8)


def plot_2d_images(img_name, img_data, label_data, pred_data):
    """
    2x2 grid layout with one empty cell.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Image", "Label", "Prediction", ""),
        specs=[
            [{"type": "image"}, {"type": "image"}],
            [{"type": "image"}, None],  # Leave (2,2) empty
        ],
    )

    # Normalize & add 3 grayscale images to 3 slots
    fig.add_trace(go.Image(z=normalize_uint8(img_data)), row=1, col=1)
    fig.add_trace(go.Image(z=normalize_uint8(label_data)), row=1, col=2)
    fig.add_trace(go.Image(z=normalize_uint8(pred_data)), row=2, col=1)

    fig.update_layout(
        title=f"{img_name} 2D Images",
        width=600,
        height=600,
        margin=dict(t=50, b=10),
        font=dict(size=14),
    )

    return fig

def plot_image_label_pred(title, image, label, pred):
    """
    Plot image, label, and prediction side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.cpu().numpy(), cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(label.cpu().numpy(), cmap="gray")
    axes[1].set_title("Label")
    axes[2].imshow(pred.cpu().numpy(), cmap="gray")
    axes[2].set_title("Prediction")

    # Add title to the figure if provided
    # if title:
    fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()
    
    return fig
    # # Save the plot to a file
    # plot_path = f"image_label_pred_{title}.png" if title else "image_label_pred.png"
    # plot_path = os.path.join(
    #     "/home/yb107/cvpr2025/DukeDiffSeg/outputs/medsegdiff/images", plot_path
    # )
    # plt.savefig(plot_path)


def get_slices_fig(img_data, img_name, n_every, volume=None, unique_values=None):
    if type(img_data) == torch.Tensor:
        img_data = img_data.cpu().numpy()

    if len(img_data.shape) == 5 or len(img_data.shape) == 4:
        img_data = np.squeeze(img_data)

    if len(img_data.shape) == 4:
        img_data = np.argmax(img_data, axis=0, keepdims=False)

    # logging.info(f"Shape of img_data: {img_data.shape}")

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(k / n_every) * np.ones(img_data[:, :, 0].shape),
                    surfacecolor=img_data[:, :, k],
                    colorscale="Viridis",
                ),
                name=str(k),
            )
            for k in range(1, img_data.shape[2], n_every)
        ]
    )
    fig.add_trace(
        go.Surface(
            z=0 * np.ones(img_data[:, :, 0].shape),
            surfacecolor=img_data[:, :, 0],
            colorscale="Viridis",
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=f"{img_name} 3D Volume Slices",
        width=800,
        height=800,
        scene=dict(
            zaxis=dict(range=[-0.1, img_data.shape[2] / n_every], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    return fig


def get_volumetric_fig(
    img_data,
    img_name,
    n_every,
    colorscale=None,
    label_of_interest=1,
    max_label_value=16,
    crop_zeros=True,
):
    if type(img_data) == torch.Tensor:
        img_data = img_data.cpu().numpy()

    if len(img_data.shape) == 5 or len(img_data.shape) == 4:
        img_data = np.squeeze(img_data)

    if len(img_data.shape) == 4:
        img_data = np.argmax(img_data, axis=0, keepdims=False)

    if type(n_every) == int:
        n_every = [n_every, n_every, n_every]

    # crop zeros
    slices = np.where(img_data > 0)
    if (not len(slices[0]) == 0) and crop_zeros:
        img_data = img_data[
            slices[0].min() : slices[0].max() + 1,
            slices[1].min() : slices[1].max() + 1,
            slices[2].min() : slices[2].max() + 1,
        ]
    else:
        raise ValueError("No non-zero values found in the image")
        # img_data = img_data[0:img_data.shape[0]//4, 0:img_data.shape[1]//4, 0:img_data.shape[2]//4]

    img_data = img_data / img_data.max()
    img_data = img_data[:: n_every[0], :: n_every[1], :: n_every[2]]
    X, Y, Z = np.mgrid[
        0 : img_data.shape[0], 0 : img_data.shape[1], 0 : img_data.shape[2]
    ]

    if colorscale is None:
        colorscale = {
            0: "rgba(0,0,0,0)",
            1: "rgba(255,0,0,1)",
            2: "rgba(255,255,0,0.8)",
            3: "rgba(255,0,0,1)",
            4: "rgba(0,255,0,0.8)",
            5: "rgba(255,255,0,0.8)",
            6: "rgba(255,0,0,1)",
            7: "rgba(0,255,255,0.8)",
            8: "rgba(255,0,255,0.8)",
            9: "rgba(255,255,0,0.8)",
            10: "rgba(0,255,0,0.8)",
            11: "rgba(255,0,0,1)",
            12: "rgba(0,255,255,0.8)",
            13: "rgba(0,0,255,0.8)",
            14: "rgba(255,0,0,1)",
            15: "rgba(255,0,0,1)",
            16: "rgba(255,0,255,0.8)",
        }
        if label_of_interest is not None:
            colorscale[label_of_interest] = "rgba(0,0,255,1)"
        colorscale = [[i / max_label_value, colorscale[i]] for i in colorscale]

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=img_data.flatten(),
            isomin=0,
            isomax=1,
            opacity=1,  # needs to be small to see through all surfaces
            surface_count=17,  # needs to be a large number for good volume rendering
            colorscale=colorscale,
        )
    )

    return fig


def get_two_mid_slices_fig(img_data, img_name="Image", label_of_interest=None):

    if type(img_data) == torch.Tensor:
        img_data = img_data.cpu().numpy()

    if len(img_data.shape) == 5 or len(img_data.shape) == 4:
        img_data = np.squeeze(img_data)

    if len(img_data.shape) == 4:
        img_data = np.argmax(img_data, axis=0, keepdims=False)

    # make 3 subplots, and plot the middle slice of each dimension
    fig = make_subplots(rows=1, cols=2, subplot_titles=("X", "Y", "Z"))

    if label_of_interest:
        masked_img = img_data == label_of_interest

        slices = np.where(masked_img > 0)
        x_mid = slices[0].min() + (slices[0].max() - slices[0].min()) // 2
        y_mid = slices[1].min() + (slices[1].max() - slices[1].min()) // 2
    else:
        x_mid = img_data.shape[0] // 2
        y_mid = img_data.shape[1] // 2

    # get the middle slice of each dimension
    x_slice = img_data[x_mid, :, :]
    y_slice = img_data[:, y_mid, :]
    # z_slice = img_data[:, :, img_data.shape[2]//2]

    fig.add_trace(go.Heatmap(z=x_slice, colorscale="Viridis"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=y_slice, colorscale="Viridis"), row=1, col=2)
    # fig.add_trace(go.Heatmap(z=z_slice, colorscale="Viridis"), row=1, col=3)

    fig.update_layout(title=f"{img_name} Middle Slices")

    return fig


def get_big_fig(img_data, title="Image", table_data=None, n_every=5):
    """
    table_data: {"header": ["Header1", "Header2", "Header3"], "data": [[1, 4], [2, 5],[3,6]]}
    """
    if type(img_data) == torch.Tensor:
        img_data = img_data.cpu().numpy()

    if len(img_data.shape) == 5 or len(img_data.shape) == 4:
        img_data = np.squeeze(img_data)

    if len(img_data.shape) == 4:
        img_data = np.argmax(img_data, axis=0, keepdims=False)

    if table_data is None:
        table_data = {
            "header": ["Key", "Value"],
            "data": [
                ["Tag", "Shape"],
                ["Value", img_data.shape],
            ],
        }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("3D Volume Slices", "Details", "Mid X Slice", "Mid Y Slice"),
        specs=[
            [{"type": "volume"}, {"type": "table"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )

    vol_fig = get_volumetric_fig(img_data, title, n_every)
    fig.add_trace(vol_fig.data[0], row=1, col=1)

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=table_data["header"]),
                cells=dict(values=table_data["data"]),
            )
        ]
    )
    fig.add_trace(table_fig.data[0], row=1, col=2)

    x_slice = img_data[img_data.shape[0] // 2, :, :]
    y_slice = img_data[:, img_data.shape[1] // 2, :]

    fig.add_trace(go.Heatmap(z=x_slice, colorscale="Viridis"), row=2, col=1)
    fig.add_trace(go.Heatmap(z=y_slice, colorscale="Viridis"), row=2, col=2)

    fig.update_layout(title=title, width=1000, height=1000)

    return fig


class UniqueValuesNumberMetric(Metric):
    def __init__(self, output_transform=None):
        self.output_transform = output_transform

    def __call__(self, y_pred):
        if self.output_transform is not None:
            y_pred = self.output_transform(y_pred)
        unique_val = torch.unique(y_pred).shape[0]
        return unique_val

class OutputVolumeMetric(CumulativeIterationMetric):
    def __init__(self, reduction=None):
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, y_pred):
        return torch.sum(y_pred[y_pred > 0]) / 1000


def get_slurm_cpus():
    """Get the number of CPUs allocated by SLURM."""
    # SLURM_CPUS_PER_TASK indicates the number of CPUs allocated per task
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        return int(cpus_per_task)

    # Alternatively, use SLURM_JOB_CPUS_PER_NODE to get the number of CPUs per node
    cpus_per_node = os.getenv("SLURM_JOB_CPUS_PER_NODE")
    if cpus_per_node is not None:
        return int(cpus_per_node)

    # Default to multiprocessing.cpu_count() if not running under SLURM or as a fallback
    return multiprocessing.cpu_count()


def resample_image_with_sitk(
    fpath: Path = None,
    img: sitk.Image = None,
    output_like: sitk.Image = None,
    method="nearest",
    out_spacing=None,
    save=False,
    output_dir=None,
    output_filename=None,
    out_orientation="RAS",
    *args,
    **kwargs,
) -> sitk.Image:
    """
    This function is rewritten to fix the issues with the original resample_image function that messed up Orientation and Origin of output image.

    Args:
    fpath : Image Full Path
    method : 0 for Linear Interpolation, 1 for Nearest Neighbour Interpolation, 2 for BSpline Interpolation
    output_like : SITK Object of Original CT Image to get the original spacing and size. Provide this if you absolutely need to match with CT Image. If only need for visualization, just provide other params.
    out_spacing : Resample Size (List in x, y z in mm)

    Returns:
    resampled ITK Image
    """
    if out_spacing is None and output_like is None:
        raise ValueError("Either out_spacing or output_like must be provided")
    if out_spacing is None:
        out_spacing = output_like.GetSpacing()

    if fpath is not None:
        itk_image = sitk.ReadImage(fpath)
    elif img is not None:
        itk_image = img
    else:
        raise ValueError("Either fpath or img must be provided")
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_origin = itk_image.GetOrigin()
    out_direction = itk_image.GetDirection()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
    ]
    if output_like is not None:
        # If output_like is provided, use its metadata
        out_spacing = output_like.GetSpacing()
        out_size = output_like.GetSize()
        out_direction = output_like.GetDirection()
        out_origin = output_like.GetOrigin()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(out_direction)
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if method == "linear":
        resample.SetInterpolator(sitk.sitkLinear)
    elif method == "nearest":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif method == "bspline":
        resample.SetInterpolator(sitk.sitkBSpline)

    resampled_img = resample.Execute(itk_image)
    # If output_like is provided, set the output image's metadata to match it
    # resampled_img = sitk.DICOMOrient(
    #     resampled_img,
    #     sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
    #         out_direction
    #     ),
    # )
    resampled_img = sitk.DICOMOrient(
        resampled_img,
        out_orientation,  # makes it LPI
    )
    resampled_img.SetOrigin(out_origin)
    # Changed to remove the need of original CT image to get origin and direction. Donot know if it will work.  check previous commits.
    # 5 hrs wasted because Origin was set before Direction.
    # I don't know why but it works now, if I set Origin after Direction.
    if save:
        output_dir = Path(output_dir) 
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_filename is None:
            if fpath is None:
                raise ValueError("Output filename must be provided")
            output_filename = fpath.stem.replace(".nii.gz", "_resampled.nii.gz")
        sitk.WriteImage(resampled_img, str(output_dir / output_filename))
    return resampled_img
