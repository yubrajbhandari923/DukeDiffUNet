from monai.transforms import (
    MapTransform,
    Transform,
    LoadImage,
    Pad,
    SpatialCrop,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    SaveImage,
)
import pandas as pd
from monai.data import ITKReader, ImageReader, ImageWriter, MetaTensor
from typing import Sequence, Union
from os import PathLike
import os
import glob
from pathlib import Path
from monai.data.meta_tensor import MetaTensor

from skimage.morphology import erosion, dilation, ball
from monai.handlers import TensorBoardImageHandler

import SimpleITK as sitk
import numpy as np
import torch
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from .visualize import Visualizer, render3d

import random
from collections import deque

import SimpleITK as sitk
import numpy as np
import logging

import gc

logging.getLogger(__name__)


class LogImgShaped(MapTransform):
    def __init__(
        self,
        keys,
        msg="Transform",
        log_unique=False,
        log_volumes=False,
        *args,
        **kwargs,
    ):
        self.msg = msg
        self.log_unique = log_unique
        self.log_volumes = log_volumes
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if not self.log_unique:
                logging.info(f"{self.msg} {key} Shape: {data[key].shape}")
            else:
                logging.info(
                    f"{self.msg} {key} Shape: {data[key].shape} Unique: {np.unique(data[key])}"
                )
            if self.log_volumes:
                for idx in np.unique(data[key]):
                    if idx == 0:
                        continue
                    logging.info(
                        f"{self.msg} {key} Volume {idx}: {np.sum(data[key] == idx)}"
                    )

        return data


class DropStructured(MapTransform):
    def __init__(
        self,
        keys,
        label_index=-1,
        background_index=0,
        random_drop_num=-1,
        keep_index=None,
    ):
        """
        If label_index is -1, then all labels except keep_index will be dropped.
        If keep_index is None, then all labels except label_index will be dropped.
        If random_drop_num is -1, then all labels will be dropped.
        If random_drop_num is 0, then no labels will be dropped.
        If random_drop_num is greater than 0, then random_drop_num labels will be dropped, excluding keep index.
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

        if label_index is None and keep_index is None:
            raise ValueError("Either label_index or keep_index must be provided")

        if label_index == -1:
            label_index = [-1]

        self.label_index = label_index
        self.keep_index = keep_index

        self.background_index = background_index
        self.random_drop_num = random_drop_num

    def __call__(self, data):
        for key in self.keys:

            # CHeck if label_index for out of range
            if not len(self.label_index) > 0:
                self.label_index = [-1]

            if self.label_index[0] == -1:
                self.label_index = np.unique(data[key])
                self.label_index = self.label_index[
                    self.label_index != self.background_index
                ]
                if self.keep_index is not None:
                    # remove the keep index from the label index
                    if not isinstance(self.keep_index, (list, tuple)):
                        self.keep_index = [self.keep_index]

                    self.label_index = [
                        i for i in self.label_index if i not in self.keep_index
                    ]

            if self.random_drop_num == 0:
                drop_structures = []
            elif self.random_drop_num > 0:
                drop_structures = np.random.choice(
                    self.label_index, self.random_drop_num, replace=False
                )
            else:
                drop_structures = self.label_index

            # logging.info(f"DropStructured: {key} Dropping {data[key]}")
            data[key] = data[key].to(torch.int32)

            for label in drop_structures:
                if label == 0:
                    continue
                data[key][data[key] == label] = self.background_index

        return data


class ResidualStructuresLabeld(MapTransform):
    def __init__(self, input_key, label_key, background_index=0):
        """Take all the structures in the input_key and remove them from the label_key. To get residual structures as label to predict."""
        self.input_key = input_key
        self.label_key = label_key
        self.background_index = background_index

    def __call__(self, data):
        input_img = data[self.input_key]
        label_img = data[self.label_key]

        for idx in np.unique(input_img):
            if idx == 0:
                continue
            label_img[label_img == idx] = self.background_index

        data[self.label_key] = label_img
        return data


class CombineStructuresd(MapTransform):
    def __init__(self, keys, label_to_structure, structure_to_label):
        """
        Args:
            keys: keys to combine the structures
            label_to_structure: function mapping label to structure name
            structure_to_label: function mapping structure name to label
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

        self.label_to_structure = label_to_structure
        self.structure_to_label = structure_to_label

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img = img.to(torch.int32)

            for idx in np.unique(img):
                if idx == 0:
                    continue
                structure = self.label_to_structure(idx)
                label = self.structure_to_label(structure)
                img[img == idx] = label

            data[key] = img

        return data


class Uniond(MapTransform):
    def __init__(self, key, label_key):
        self.key = key
        self.label_key = label_key

    def __call__(self, data):
        # for key in self.keys:
        data[self.key][data[self.label_key] != 0] = data[self.label_key][
            data[self.label_key] != 0
        ]
        # data[self.label_key][data[key] != 0] = data[key][data[key] != 0]
        data[self.label_key] = data[self.key]  # Just to make sure it is updated
        return data


class IndividualMasksToCombinedMask(Transform):
    def __init__(
        self,
        value_mapping,
        random_drop=-1,
        structures=None,
        match_shape_if_needed=True,
        allow_missing_structures=False,
    ):
        if isinstance(value_mapping, dict):
            value_mapping = value_mapping.get

        self.value_mapping = value_mapping
        self.random_drop = random_drop
        self.structures = structures
        self.match_shape_if_needed = match_shape_if_needed
        self.allow_missing_structures = allow_missing_structures

    def __call__(self, folder: str):
        if type(folder) is str:
            folder = Path(folder)

        structures = os.listdir(folder) if self.structures is None else self.structures

        if self.random_drop > 0:
            structures = random.sample(structures, len(structures) - self.random_drop)

        meta = None

        # structures = structures[:20]
        # combined_mask = np.array()
        logging.info(f"==============================================================")
        for idx, structure in enumerate(structures):
            structure_path = folder / structure

            if not os.path.isfile(structure_path):
                if os.path.exists(structure_path.with_suffix(".nii.gz")):
                    structure_path = structure_path.with_suffix(".nii.gz")
                else:
                    if self.allow_missing_structures:
                        logging.info(
                            f"\n\n Structure {structure} not found in {folder}. Skipping. \n\n"
                        )
                        continue

                    raise ValueError(
                        f"Structure {structure} not found in {structure_path}"
                    )

            try:
                structure_img = LoadImage()(str(structure_path))
            except Exception as e:
                raise ValueError(f"Error loading {structure_path}: {e}")

            structure_data = structure_img.get_array()
            structure_data = np.squeeze(structure_data).astype(bool)

            if idx == 0:
                meta = structure_img.meta
                affine = structure_img.affine
                combined_mask = np.zeros_like(structure_data, dtype=np.int16)

            # Check if affine are the same for structure_img and meta
            # assert bool(
            #     (structure_img.affine == affine).all()
            # ), f"Affine mismatch {affine} vs {structure_img.affine}"

            # if not bool((structure_img.affine == affine).all()):
            #     logging.info(f"Affine mismatch {affine} vs {structure_img.affine}")

            # logging.info(
            #     f"Structure: {structure} Value: {self.value_mapping(structure)}"
            # )
            # logging.info(
            #     f"Shape: {structure_data.shape} Unique: {np.unique(structure_data)}"
            # )

            if combined_mask.shape != structure_data.shape:
                logging.info(
                    f"Shape mismatch: {combined_mask.shape} != {structure_data.shape}"
                )
                if self.match_shape_if_needed:
                    combined_img = MetaTensor(combined_mask, meta=meta)
                    tmp_data = {
                        "combined": combined_img,
                        "structure": structure_img,
                    }
                    tmp_data = MatchShapeByPadd(keys=["combined", "structure"])(
                        tmp_data
                    )
                    combined_mask = tmp_data["combined"].squeeze().get_array()
                    structure_data = (
                        tmp_data["structure"].squeeze().get_array().astype(bool)
                    )
                    logging.info(
                        f"Shape after match: {combined_mask.shape} == {structure_data.shape}"
                    )
                else:
                    raise ValueError(
                        f"Shape mismatch: {combined_mask.shape} != {structure_data.shape}"
                    )

            combined_mask[structure_data] = self.value_mapping(
                structure.replace(".nii.gz", "")
            )
            # logging.info(f"Structure: {structure} Value: {self.value_mapping(structure)}, combined_mask: {np.unique(combined_mask)}")

        # combined_mask = combined_mask.astype(np.int16)
        # logging.info(f"Combine Unique: {np.unique(combined_mask)}")

        combined_mask = MetaTensor(combined_mask, meta=meta)
        combined_mask.meta["filename_or_obj"] = folder.name
        return combined_mask


class IndividualMasksToCombinedMaskd(MapTransform):

    def __init__(
        self,
        keys,
        value_mapping,
        random_drop_num=-1,
        structures=None,
        match_shape_if_needed=True,
        allow_missing_structures=False,
    ):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.value_mapping = value_mapping
        self.random_drop_num = random_drop_num
        self.structures = structures
        self.match_shape_if_needed = match_shape_if_needed
        self.allow_missing_structures = allow_missing_structures

    def __call__(self, data):
        for key in self.keys:
            folder = data[key]
            combined_mask = IndividualMasksToCombinedMask(
                self.value_mapping,
                self.random_drop_num,
                self.structures,
                self.match_shape_if_needed,
                self.allow_missing_structures,
            )(folder)
            data[key] = combined_mask
        return data


class ConvertMapping(Transform):
    def __init__(self, original_map, new_map):
        self.original_map = original_map
        self.new_map = new_map

    def __call__(self, img):

        # make the dtype int16
        img = img.to(torch.int32)

        for label in np.unique(img):
            if label == 0:
                continue

            try:
                img[img == label] = self.new_map(self.original_map(label))
            except:
                raise ValueError(f"Error converting label {label}")
        return img


class ConvertMappingd(MapTransform):
    def __init__(self, keys, original_map, new_map):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.original_map = original_map
        self.new_map = new_map

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img = ConvertMapping(self.original_map, self.new_map)(img)
            data[key] = img
        return data


class CopyKeyd(MapTransform):
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        # data[self.target_key] = data[self.source_key]
        if isinstance(data[self.source_key], MetaTensor):
            data[self.target_key] = data[self.source_key].clone()
        elif isinstance(data[self.source_key], str):
            data[self.target_key] = data[self.source_key]
        else:
            data[self.target_key] = data[self.source_key].clone()
        return data


class Visualize(Transform):
    def __init__(self, output_path, filename=None, **kwargs):
        self.output_path = output_path
        self.kwargs = kwargs
        self.filename = filename

    def __call__(self, img):
        if self.filename is not None:
            file_name = self.filename
        else:
            file_name = (
                img.meta["filename_or_obj"].replace(".nii.gz", "").split("/")[-1]
            )

        render3d(img, self.output_path, file_name=file_name, **self.kwargs)

        return img


class Visualized(MapTransform):

    def __init__(self, keys, output_path, filename=None, **kwargs):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        if filename is not None:
            if not isinstance(filename, (list, tuple)):
                filename = [filename]

        self.keys = keys
        self.output_path = output_path
        self.kwargs = kwargs
        self.filename = filename

    def __call__(self, data):
        for idx, key in enumerate(self.keys):
            img = data[key]

            if self.filename is not None:
                file_name = self.filename[idx]
                if file_name == "":
                    file_name = (
                        img.meta["filename_or_obj"]
                        .replace(".nii.gz", "")
                        .split("/")[-1]
                    )
            else:
                file_name = (
                    img.meta["filename_or_obj"].replace(".nii.gz", "").split("/")[-1]
                )

            render3d(img, self.output_path, file_name=file_name, **self.kwargs)
        return data


class MatchShapeByPadd(MapTransform):
    def __init__(self, keys, add=None):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.add = add

    def __call__(self, data):
        maximum = np.array([0, 0, 0])
        for k in self.keys:
            data[k] = data[k].squeeze()
            img = data[k]

            if len(img.shape) != 3:
                raise ValueError(f"Image shape {img.shape} not supported")

            maximum = np.maximum(maximum, img.shape)

        if self.add is not None:
            maximum += self.add

        data = EnsureChannelFirstd(keys=self.keys)(data)
        for k in self.keys:
            logging.info(f"Key: {k}, Shape: {data[k].shape}")
        logging.info(f"Maximum shape: {maximum}, Keys: {self.keys}")
        data = ResizeWithPadOrCropd(keys=self.keys, spatial_size=maximum)(data)

        return data


class AddTensord(MapTransform):
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        source_img = data[self.source_key].squeeze().numpy()
        target_img = data[self.target_key].squeeze().numpy()
        data[self.target_key].set_array(target_img + source_img)
        return data


class DropKeysd(MapTransform):
    def __init__(self, keys):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            del data[key]

        torch.cuda.empty_cache()
        gc.collect()

        return data


class LogDeviced(MapTransform):
    def __init__(self, keys, msg=""):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.msg = msg

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            logging.info(
                f"Key: {key} Device: {img.device}, Shape: {img.shape}, {self.msg}"
            )
        return data


class LogDiced(MapTransform):
    def __init__(
        self,
        target_key,
        template_key,
        target_label=None,
        template_label=None,
        msg="Dice Score",
    ):
        self.target_key = target_key
        self.template_key = template_key
        self.target_label = target_label
        self.template_label = template_label
        self.msg = msg

    def dice(self, img1, img2):
        if len(img1.unique()) > 2 or len(img2.unique()) > 2:
            logging.warning(
                f"Unique values in img1: {img1.unique()}, img2: {img2.unique()}"
            )

            for i in img1.unique():
                if i not in img2.unique():
                    img1[img1 == i] = 0
            for i in img2.unique():
                if i not in img1.unique():
                    img2[img2 == i] = 0

            total_dice = 0
            for i in img1.unique():
                if i == 0:
                    continue
                dice_score = self.dice(img1 == i, img2 == i)
                logging.info(f"Dice score for {i}: {dice_score}")
                total_dice += dice_score

            return total_dice / len(img1.unique())

        return 2 * (img1 * img2).sum() / (img1.sum() + img2.sum())

    def __call__(self, data):
        target = data[self.target_key]
        template = data[self.template_key]

        if self.target_label is not None:
            target = target == self.target_label
        if self.template_label is not None:
            template = template == self.template_label

        logging.info(f"{self.msg}: {self.dice(target.squeeze(), template.squeeze())}")

        return data


class SaveLabelsSeperate(MapTransform):
    def __init__(self, keys, output_path_key, class_map, labels=None):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.labels = labels
        self.output_path_key = output_path_key
        self.class_map = class_map

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            save_path = data[self.output_path_key]

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if self.labels is None:
                self.labels = img.unique()

            for label in self.labels:
                if label == 0:
                    continue

                img_copy = img.clone()
                img_copy[img_copy != label] = 0
                img_copy[img_copy == label] = 1
                img_copy.meta["filename_or_obj"] = os.path.join(
                    save_path, f"{self.class_map[label.item()]}.nii.gz"
                )

                SaveImage(
                    output_dir=save_path, output_postfix="", separate_folder=False
                )(img_copy)

        return data
