import os
from pathlib import Path
import numpy as np
import nibabel as nib
import numpy as np
import vtk
from vtk.util import numpy_support
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from fury import window
import matplotlib.pyplot as plt

from monai.data import MetaTensor

import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

from xvfbwrapper import Xvfb

np.random.seed(100)
random_colors = np.random.rand(200, 5)


def colors_map(filename):
    colors = {
        "bones": [0.9568627450980393, 0.8862745098039215, 0.7764705882352941, 1.0],
        "eye": [0.9568627450980393, 0.8862745098039215, 0.7764705882352941, 1.0],
        "muscles": [0.9294117647058824, 0.5333333333333333, 0.4470588235294118, 1.0],
        "dia": [0.9294117647058824, 0.5333333333333333, 0.4470588235294118, 1.0],
        "autochthon": [0.9294117647058824, 0.5333333333333333, 0.4470588235294118, 1.0],
        "adrenal": [0.9607843137254902, 0.807843137254902, 0.7137254901960784, 1.0],
        "arteries": [1, 0, 0, 1.0],
        "veins": [0, 0, 1, 1.0],
        "colon": [0.9921568627450981, 0.6, 0.3686274509803922, 1.0],  # large intestine
        "rectum": [0.9921568627450981, 0.6, 0.3686274509803922, 1.0],  # large intestine
        "duodenum": [0.7333333333333333, 0.5843137254901961, 0.788235294117647, 1.0],
        "esophagus": [0.9882352941176471, 0.4980392156862745, 0.49019607843137253, 1.0],
        "gallbladder": [0.16470588235294117, 0.6235294117647059, 0.0, 1.0],
        # "heart": [0.9, 0.9, 0.9, 1.0],
        "kidney": [0.7529411764705882, 0.32941176470588235, 0.2980392156862745, 1.0],
        "liver": [0.8588235294117647, 0.3058823529411765, 0.3254901960784314, 1.0],
        "lung": [0.9, 0.9, 0.9, 1.0],
        "pancreas": [0.9490196078431372, 0.7647058823529411, 0.25882352941176473, 1.0],
        "spinal": [0.788235294117647, 0.4549019607843137, 0.6039215686274509, 1.0],
        "spleen": [0.788235294117647, 0.4549019607843137, 0.6039215686274509, 1.0],
        "stomach": [1.0, 0.49019607843137253, 0.6313725490196078, 1.0],
        "trachea": [0.9, 0.9, 0.9, 1.0],
        "viseceral": [0.9, 0.9, 0.9, 1.0],
        "subcutaneous": [0.9, 0.9, 0.9, 1.0],
        "skin": [0.8235294117647058, 0.596078431372549, 0.5686274509803921, 1.0],
        "body": [0.9, 0.9, 0.9, 1.0],
        "gland": [0.7411764705882353, 0.43137254901960786, 0.2784313725490196, 1.0],
        # "muscles": [0.5333333333333333, 0.20784313725490197, 0.1843137254901961, 1.0],
        "brain": [0.9647058823529412, 0.7098039215686275, 0.6901960784313725, 1.0],
        "brainstem": [0.8901960784313725, 0.5215686274509804, 0.4392156862745098, 1.0],
        # "small": [
        #     0.7843137254901961,
        #     0.1843137254901961,
        #     0.24705882352941178,
        #     1.0,
        # ],  # small intestine
        "small": [63 / 255, 191 / 255, 191 / 255, 1.0],
        "default": [0, 1, 0, 1.0],
        "lips": [0.996078431372549, 0.8352941176470589, 0.7843137254901961, 1.0],
        "ribc": [190 / 255, 190 / 255, 190 / 255, 1.0],
    }
    colors["parotid"] = colors["gland"]
    colors["submandibular"] = colors["gland"]

    try:
        keyword = filename.split(os.sep)[-1].split("_")[0].replace(".obj", "").lower()
        keyword = keyword.split("1")[0]
    except:
        raise ValueError(f"Warning: color not found for {filename}")

    bones = [
        "hip",
        "mandible",
        "rib",
        "skull",
        "spine",
        "sternum",
        "teeth",
        "vertebrae",
        "clavicle",
        "pelvis",
        "scapula",
        "femur",
        "fibula",
        "humerus",
        "patella",
        "radius",
        "tibia",
        "ulna",
        "brachial",
        "clavicula",
        "carpal",
        "fingers",
        "metacarpal",
        "metatarsal",
        "patella",
        "radius",
        "tarsal",
        "tibia",
        "toes",
        "ulna",
        "costal",
        "bones",
        "hyoid",
        "cart",
        "sinus",
        "trapezium",
        "trapezoid",
        "capitate",
        "hamate",
        "lunate",
        "scaphoid",
        "pisiform",
        "pisform",
        "triquetrum",
        "sacrum",
        "metacarpal",
        "phalanx",
        "triquetrum",
        "calceneus",
        "cuboid",
        "navicular",
        "tanus",
        "cuneiform",
    ]
    muscles = ["autochthon", "gluteus", "iliopsoas", "oral", "musc"]
    if keyword in colors.keys():
        return colors[keyword]
    if keyword == "heart":
        full_name = filename.split("/")[-1].replace(".obj", "").lower()
        map = {
            "heart_atrium_left": [
                0.7843137254901961,
                0.07450980392156863,
                0.07058823529411765,
                1.0,
            ],
            "heart_atrium_right": [
                0.3176470588235294,
                0.4235294117647059,
                0.6470588235294118,
                1.0,
            ],
            "heart_ventricle_left": [
                0.8980392156862745,
                0.18823529411764706,
                0.11764705882352941,
                1.0,
            ],
            "heart_ventricle_right": [
                0.36470588235294116,
                0.4745098039215686,
                0.6784313725490196,
                1.0,
            ],
            "heart_myocardium": [
                0.9882352941176471,
                0.792156862745098,
                0.7490196078431373,
                1.0,
            ],
        }
        return map[full_name]
    if keyword == "lung":
        full_name = filename.split("/")[-1].replace(".obj", "").lower()
        map = {
            "lung_lower_lobe_left": [
                0.6588235294117647,
                0.7568627450980392,
                0.9333333333333333,
                1.0,
            ],
            "lung_lower_lobe_right": [
                0.6588235294117647,
                0.7568627450980392,
                0.9333333333333333,
                1.0,
            ],
            "lung_upper_lobe_left": [1.0, 0.807843137254902, 0.8, 1.0],
            "lung_upper_lobe_right": [1.0, 0.807843137254902, 0.8, 1.0],
            "lung_middle_lobe_right": [
                0.8313725490196079,
                0.6862745098039216,
                0.8588235294117647,
                1.0,
            ],
            "lung_trachea_bronchia": [
                0.9019607843137255,
                0.8980392156862745,
                0.9686274509803922,
                1.0,
            ],
            "lung_vessels": [
                0.8666666666666667,
                0.6627450980392157,
                0.7058823529411765,
                1.0,
            ],
        }
        return map[full_name]

    if keyword in bones:
        return colors["bones"]
    if keyword in muscles:
        return colors["muscles"]
    if keyword in ["pulmonary", "bronchial", "artery", "aorta", "optic"]:
        return colors["arteries"]
    if keyword in ["portal", "inferior"]:
        return colors["veins"]

    # Open a file and write the color
    with open("tmp", "a") as myfile:
        myfile.write(f"Warning: color not found for {filename}; keyword: {keyword}\n")

    return colors["default"]


class Visualizer:
    """
    Usuage:
        vis = Visualizer(...)
        vis.generate_preview(input_image, output_path, file_name)
    """

    def __init__(
        self,
        roi_groups,
        class_map,
        smoothing=0,
        n_every=1,
        flipped=False,
        window_size=(1200, 800),
        subject_width=400,
        write_file_name=True,
        close_up=False,
    ):
        """
        Args:
            smoothing (int): The smoothing factor for the image.
            roi_groups (list): The roi groups for the image. Eg: [["muscles"], ["bones"], ["organs"]]
            class_map (dict): The class map for the image. {"muscles": 1, "bones": 2, "organs": 3}
            smoothed (int): The smoothing factor for the image. Default is 0. More the value, more slower.
            n_every (int): The factor to reduce the image. Default is 1. More the value, more faster.
            flipped (bool): Whether to flip the image. Default is False.
            window_size (tuple): The window size for the image. Default is (1200, 800). Try out different values.
            subject_width (int): The subject width for the image. Default is 400. Try out different values.
            write_file_name (bool): Whether to write the file name on the image. Default is True.
            close_up (bool): Whether to zoom in the image. Default is False.
        """

        self.smoothing = smoothing
        self.roi_groups = roi_groups
        self.class_map = class_map
        self.flipped = flipped
        self.n_every = n_every
        self.write_file_name = write_file_name

        self.window_size = window_size
        self.subject_width = subject_width
        self.closeup = close_up

    def generate_preview(self, input_image, output_path, file_name=None):
        # do not set random seed, otherwise can not call xvfb in parallel, because all generate same tmp dir
        if (
            type(input_image) is str
            or type(input_image) is Path
            or isinstance(input_image, os.PathLike)
        ):
            if os.path.isdir(input_image):
                for file in os.listdir(input_image):
                    if file.endswith(".nii.gz"):
                        logging.info(f"Generating preview for {file}")
                        with Xvfb() as xvfb:
                            return self.plot_subject(
                                output_path,
                                os.path.join(input_image, file),
                                self.roi_groups,
                                file_name=file,
                            )
            else:
                with Xvfb() as xvfb:
                    return self.plot_subject(
                        output_path,
                        input_image,
                        self.roi_groups,
                        file_name=file_name,
                    )
        else:
            with Xvfb() as xvfb:
                return self.plot_subject(
                    output_path,
                    input_image,
                    self.roi_groups,
                    file_name=file_name,
                )

    def plot_subject(self, output_path, input, roi_groups, file_name=None):
        """
        Plot a single subject with all rois in one image.
        Args:
            output_path (str): The output path for the image, must end with .png.
            input (Nifiti1Image / str / Path): The input path for the image or the nifiti Image. Can't support np.ndarray because of misssing affine information.
            smoothing (int): The smoothing factor for the image.
        """
        scene = window.Scene()
        showm = window.ShowManager(scene, size=self.window_size, reset_camera=False)
        showm.initialize()

        if os.path.isdir(output_path):
            if not file_name:
                raise ValueError(
                    "file_name must be provided if output_path is a directory"
                )

            output_path = os.path.join(output_path, file_name + ".png")

        roi_data = None
        # Check if input_path is a folder with multiple nifti files
        if type(input) is str or type(input) is Path or isinstance(input, os.PathLike):
            input = nib.load(input)

        if type(input) is nib.nifti1.Nifti1Image:
            # This code might need some refactoring
            roi_data = input.get_fdata()  # numpy array
            affine = input.affine
        elif type(input) is MetaTensor:
            roi_data = input.squeeze().cpu().numpy()
            affine = input.affine.cpu().numpy()

        else:
            raise ValueError(
                f"{input} is not a nifti file nor a folder with nifti structures. It is {type(input)}"
            )

        if roi_data is None:
            raise ValueError("No data found in the input file.")

        if roi_data.ndim != 3:
            raise ValueError(
                f"Only 3D arrays are currently supported. Found {roi_data.shape} shape array."
            )

        if not self.closeup:
            roi_data[0, 0, 0] = 1
            roi_data[-1, -1, -1] = 1

        if self.n_every > 1:
            roi_data = roi_data[:: self.n_every, :: self.n_every, :: self.n_every]

        for idx, roi_group in enumerate(roi_groups):
            x = idx * (self.window_size[0] // len(roi_groups))
            y = 0

            self.plot_roi_group(scene, roi_group, x, y, roi_data, affine)

        ## After adding all the actors
        scene.projection(proj_type="parallel")
        scene.reset_camera_tight(
            margin_factor=1.02
        )  # need to do reset_camera=False in record for this to work in

        # window.record(
        #     scene, size=self.window_size, out_path=output_path, reset_camera=False
        # )  # , reset_camera=False
        img_data = window.snapshot(scene, size=self.window_size, offscreen=True)

        # Save the image with matplotlib and filename at the bottom
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_data)
        ax.axis("off")
        if self.write_file_name and file_name:
            ax.text(
                5, -10, file_name, fontsize=9, color="white", backgroundcolor="black"
            )
        fig.savefig(output_path, bbox_inches="tight")

        logging.info(f"Preview Generated at {output_path}")
        scene.clear()
        return fig

    def plot_roi_group(self, scene, rois, x, y, roi_data, affine):
        """
        TODO: Fix this
        Plot all rois of a group in one image.
        """
        classname_2_idx = self.class_map

        for idx, roi in enumerate(rois):
            # color = random_colors[idx]
            color = colors_map(roi)[:3]

            if type(classname_2_idx) is dict:
                classname_2_idx = classname_2_idx.get

            data = roi_data == classname_2_idx(roi)

            if data.max() > 0:  # empty mask [Yubraj: should it be non empty mask?]
                affine[:3, 3] = 0  # make offset the same for all subjects
                cont_actor = self.plot_mask(
                    data,
                    affine,
                    x,
                    y,
                    color=color,
                    opacity=1,
                )
                scene.add(cont_actor)

    def plot_mask(
        self,
        mask_data,  # roi_data
        affine,
        x_current,
        y_current,
        color=[1, 0.27, 0.18],
        opacity=1,
    ):
        """
        color: default is red
        """
        # 3D Bundle
        mask = mask_data
        # mask = mask.transpose(0, 2, 1)

        # Probably needed for vltk
        mask = np.swapaxes(mask, 1, 2)

        if self.flipped:
            mask = np.flip(mask, axis=2)

        mask = mask[::-1, :, :]

        cont_actor = self.contour_from_roi_smooth(
            mask, affine=affine, color=color, opacity=opacity, smoothing=self.smoothing
        )
        cont_actor.SetPosition(x_current, y_current, 0)
        return cont_actor

    def contour_from_roi_smooth(
        self, data, affine=None, color=np.array([1, 0, 0]), opacity=1, smoothing=0
    ):
        """Generates surface actor from a binary ROI.
        Code from dipy, but added awesome smoothing!
        Parameters
        ----------
        data : array, shape (X, Y, Z)
            An ROI file that will be binarized and displayed.
        affine : array, shape (4, 4)
            Grid to space (usually RAS 1mm) transformation matrix. Default is None.
            If None then the identity matrix is used.
        color : (1, 3) ndarray
            RGB values in [0,1].
        opacity : float
            Opacity of surface between 0 and 1.
        smoothing: int
            Smoothing factor e.g. 10.
        Returns
        -------
        contour_assembly : vtkAssembly
            ROI surface object displayed in space
            coordinates as calculated by the affine parameter.
        """
        major_version = vtk.vtkVersion.GetVTKMajorVersion()

        if data.ndim != 3:
            raise ValueError("Only 3D arrays are currently supported.")
        else:
            nb_components = 1

        data = (data > 0) * 1
        vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
        vol = vol.astype("uint8")

        im = vtk.vtkImageData()
        if major_version <= 5:
            im.SetScalarTypeToUnsignedChar()
        di, dj, dk = vol.shape[:3]
        im.SetDimensions(di, dj, dk)
        voxsz = (1.0, 1.0, 1.0)
        # im.SetOrigin(0,0,0)
        im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
        if major_version <= 5:
            im.AllocateScalars()
            im.SetNumberOfScalarComponents(nb_components)
        else:
            im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, nb_components)

        # copy data
        vol = np.swapaxes(vol, 0, 2)
        vol = np.ascontiguousarray(vol)

        if nb_components == 1:
            vol = vol.ravel()
        else:
            vol = np.reshape(vol, [np.prod(vol.shape[:3]), vol.shape[3]])

        uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
        im.GetPointData().SetScalars(uchar_array)

        if affine is None:
            affine = np.eye(4)

        # Set the transform (identity if none given)
        transform = vtk.vtkTransform()
        transform_matrix = vtk.vtkMatrix4x4()
        transform_matrix.DeepCopy(
            (
                affine[0][0],
                affine[0][1],
                affine[0][2],
                affine[0][3],
                affine[1][0],
                affine[1][1],
                affine[1][2],
                affine[1][3],
                affine[2][0],
                affine[2][1],
                affine[2][2],
                affine[2][3],
                affine[3][0],
                affine[3][1],
                affine[3][2],
                affine[3][3],
            )
        )
        transform.SetMatrix(transform_matrix)
        transform.Inverse()

        # Set the reslicing
        image_resliced = vtk.vtkImageReslice()
        self.set_input(image_resliced, im)  # self.set_input(image_resliced, im)
        image_resliced.SetResliceTransform(transform)
        image_resliced.AutoCropOutputOn()

        # Adding this will allow to support anisotropic voxels
        # and also gives the opportunity to slice per voxel coordinates

        rzs = affine[:3, :3]
        zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
        image_resliced.SetOutputSpacing(*zooms)

        image_resliced.SetInterpolationModeToLinear()
        image_resliced.Update()

        # skin_extractor = vtk.vtkContourFilter()
        skin_extractor = vtk.vtkMarchingCubes()
        if major_version <= 5:
            skin_extractor.SetInput(image_resliced.GetOutput())
        else:
            skin_extractor.SetInputData(image_resliced.GetOutput())
        skin_extractor.SetValue(0, 100)

        if smoothing > 0:
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(skin_extractor.GetOutputPort())
            smoother.SetNumberOfIterations(smoothing)
            smoother.SetRelaxationFactor(0.1)
            smoother.SetFeatureAngle(60)
            smoother.FeatureEdgeSmoothingOff()
            smoother.BoundarySmoothingOff()
            smoother.SetConvergence(0)
            smoother.Update()

        skin_normals = vtk.vtkPolyDataNormals()
        if smoothing > 0:
            skin_normals.SetInputConnection(smoother.GetOutputPort())
        else:
            skin_normals.SetInputConnection(skin_extractor.GetOutputPort())
        skin_normals.SetFeatureAngle(60.0)

        skin_mapper = vtk.vtkPolyDataMapper()
        skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
        skin_mapper.ScalarVisibilityOff()

        skin_actor = vtk.vtkActor()
        skin_actor.SetMapper(skin_mapper)
        skin_actor.GetProperty().SetOpacity(opacity)
        skin_actor.GetProperty().SetColor(color[0], color[1], color[2])

        return skin_actor

    def set_input(self, vtk_object, inp):
        """Set Generic input function which takes into account VTK 5 or 6.
        Parameters
        ----------
        vtk_object: vtk object
        inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput
        Returns
        -------
        vtk_object
        Notes
        -------
        This can be used in the following way::
            from fury.utils import set_input
            poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)
        This function is copied from dipy.viz.utils
        """
        if isinstance(inp, (vtk.vtkPolyData, vtk.vtkImageData)):
            vtk_object.SetInputData(inp)
        elif isinstance(inp, vtk.vtkAlgorithmOutput):
            vtk_object.SetInputConnection(inp)
        vtk_object.Update()
        return vtk_object


def render3d(input_image, output_path, roi_groups, class_map, file_name=None, **kwargs):
    """
    Visualize the input image.
    Args:
        input_image (str / MetaTensor): The input image path. (nifti file / MetaTensor)
        output_path (str): The output path for the image, can be a directory or a file path.
        roi_groups (list): The roi groups for the image. Eg: [["muscles"], ["bones"], ["organs"]]
        class_map (dict): The class map for the image. {"muscles": 1, "bones": 2, "organs": 3}
        file_name (str): The file name for the image. Must be provided if output_path is a directory. Will also be written on the image.
    """
    vis = Visualizer(roi_groups, class_map, **kwargs)
    return vis.generate_preview(input_image, output_path, file_name)


def create_collage(images, output_path, rows, cols, width=400, padding=10):
    """
    Create a collage of images.
    Args:
        images (list): List of images to create a collage.
        output_path (str): The output path for the collage img.
        rows (int): The number of rows in the collage.
        cols (int): The number of columns in the collage.
        width (int): The width of each image.
        padding (int): The padding between each image.
    """

    imgs = [Image.open(i) for i in images]
    widths, heights = zip(*(i.size for i in imgs))

    total_width = cols * width + (cols - 1) * padding
    max_height = rows * max(heights) + (rows - 1) * padding

    collage = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    y_offset = 0

    for idx, img in enumerate(imgs):
        collage.paste(img, (x_offset, y_offset))
        x_offset += width + padding
        if x_offset >= total_width:
            x_offset = 0
            y_offset += max(heights) + padding
    collage.save(output_path)
    logging.info(f"Collage Generated at {output_path}")
    return collage
