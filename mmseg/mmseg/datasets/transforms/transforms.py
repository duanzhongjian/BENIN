# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Iterable, Sequence, Tuple, Union, Optional, List
import warnings
import cv2
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of, is_list_of
from numpy import random

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS

Number = Union[int, float]

@TRANSFORMS.register_module()
class ResizeToMultiple(BaseTransform):
    """Resize images & seg to multiple of divisor.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - pad_shape

    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self, size_divisor=32, interpolation=None):
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.
        img = results['img']
        img = mmcv.imresize_to_multiple(
            img,
            self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation
            if self.interpolation else 'bilinear')

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]

        # Align segmentation map to multiple of size divisor.
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            gt_seg = mmcv.imresize_to_multiple(
                gt_seg,
                self.size_divisor,
                scale_factor=1,
                interpolation='nearest')
            results[key] = gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size_divisor={self.size_divisor}, '
                     f'interpolation={self.interpolation})')
        return repr_str


@TRANSFORMS.register_module()
class Rerange(BaseTransform):
    """Rerange the image pixel value.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results: dict) -> dict:
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def transform(self, results: dict) -> dict:
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '\
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@TRANSFORMS.register_module()
class BioPatchCrop(BaseTransform):
    """Crop the input patch for medical image & seg.
    Required Keys:
        - img
        - gt_seg_map
    Modified Keys:
        - img
        - img_shape
        - gt_seg_map
    Args:
        crop_size (Union[int, Tuple[int, int, int]]):  Expected size after
            cropping with the format of (z, y, x). If set to an integer,
            then cropping width and height are equal to this integer.
        force_fg (bool): Cropped patch must contain foreground.
        crop_mode (str): The crop mode for biomedical patch cropping.
            Default: nnUNet
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int, int]],
                 force_fg: bool = True,
                 crop_mode: str = 'nnUNet',
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 3
        ), 'The expected crop_size is an integer, or a tuple containing three '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0 and crop_size[2] > 0
        self.crop_size = crop_size
        self.force_fg = force_fg
        self.crop_mode = crop_mode
        self.ignore_index = ignore_index

    def sample_locations(self, seg_map: np.ndarray) -> dict:
        """sample foreground locations for "nnUNet" crop_mode.
        Args:
            seg_map (np.ndarray): gt seg map
        Returns:
            dict: Coordinates of selected foreground locations
        """
        num_samples = 10000
        # at least 1% of the class voxels need to be selected,
        # otherwise it may be too sparse
        min_percent_coverage = 0.01
        rndst = np.random.RandomState(1234)
        class_locs = {}
        all_classes = np.unique(seg_map)
        for c in all_classes:
            if c == 0 or c == self.ignore_index:
                continue
            all_locs = np.argwhere(seg_map == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(
                target_num_samples,
                int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(
                len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
        return class_locs

    def generate_crop_bbox(self, results: dict) -> tuple:
        """Randomly get a crop bounding box with specific crop mode.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            tuple: Coordinates of the cropped image.
        """

        def random_generate_crop_bbox(seg_map: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.
            Args:
                seg_map (np.ndarray): Ground truth segmentation map.
            Returns:
                tuple: Coordinates of the cropped image.
            """
            margin_d = max(seg_map.shape[0] - self.crop_size[0], 0)
            margin_h = max(seg_map.shape[1] - self.crop_size[1], 0)
            margin_w = max(seg_map.shape[2] - self.crop_size[2], 0)
            offset_d = np.random.randint(0, margin_d + 1)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_z1, crop_z2 = offset_d, offset_d + self.crop_size[0]
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[1]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[2]

            return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

        seg_map = results['gt_seg_map']
        if self.force_fg:
            if self.crop_mode == 'nnUNet':
                class_locs = self.sample_locations(seg_map)
                foreground_classes = np.array(
                    [i for i in class_locs.keys() if len(class_locs[i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain
                    # foreground voxels at all
                    print('case does not contain any foreground classes: ',
                          results['img_path'])
                    crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 \
                        = random_generate_crop_bbox(seg_map)
                else:
                    selected_class = np.random.choice(foreground_classes)
                    voxels_of_that_class = class_locs[selected_class]
                    selected_voxel = voxels_of_that_class[np.random.choice(
                        len(voxels_of_that_class))]

                    margin_d = max(0,
                                   selected_voxel[0] - self.crop_size[0] // 2)
                    margin_h = max(0,
                                   selected_voxel[1] - self.crop_size[1] // 2)
                    margin_w = max(0,
                                   selected_voxel[2] - self.crop_size[2] // 2)
                    offset_d = np.random.randint(0, margin_d + 1)
                    offset_h = np.random.randint(0, margin_h + 1)
                    offset_w = np.random.randint(0, margin_w + 1)
                    crop_z1, crop_z2 = offset_d, offset_d + self.crop_size[0]
                    crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[1]
                    crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[2]
            else:
                crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 \
                    = random_generate_crop_bbox(seg_map)
                # TODO monai cropped modes.
        else:
            crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 \
                = random_generate_crop_bbox(seg_map)

        return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``
        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.
        Returns:
            np.ndarray: The cropped image.
        """
        crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if len(img.shape) == 3:
            img = img[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            assert len(img.shape) == 4
            img = img[:, crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps with specifical crop mode.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        crop_bbox = self.generate_crop_bbox(results)

        # crop the image
        img = results['img']
        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape[1:]

        # crop semantic seg
        seg_map = results['gt_seg_map']
        results['gt_seg_map'] = self.crop(seg_map, crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@TRANSFORMS.register_module()
class MedPad(BaseTransform):
    """Pad the image & segmentation map. There are three padding modes: (1) pad
    to a fixed size and (2) pad to the minimum size that is divisible by some
    number. Required Keys:
    Required Keys:
    - img
    - gt_seg_map (optional)
    Modified Keys:
    - img
    - gt_seg_map
    - img_shape
    Added Keys:
    - pad_shape
    - pad_fixed_size
    - pad_size_divisor
    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (w, h). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_val (Number | dict[str, Number], optional): Padding value for if
            the pad_mode is "constant". If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 0. If it is a dict, it should have the
            following keys:
            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=0).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.
            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self,
                 size: Optional[Tuple[int, int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: Union[Number, dict] = dict(img=0, seg=0),
                 padding_mode: str = 'constant') -> None:

        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, int):
            pad_val = dict(img=pad_val, seg=0)
        assert isinstance(pad_val, dict), 'pad_val '
        self.pad_val = pad_val

        assert size is not None or size_divisor is not None, \
            'only one of size and size_divisor should be valid'
        assert size is None or size_divisor is None

        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.padding_mode = padding_mode

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""

        pad_val = self.pad_val.get('img', 0)

        size = None
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'].shape[1], results['img'].shape[2],
                        results['img'].shape[3])
            pad_z = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_x = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            pad_y = int(np.ceil(
                size[2] / self.size_divisor)) * self.size_divisor
            size = (pad_z, pad_x, pad_y)
        elif self.size is not None:
            size = self.size

        padded_img = self._to_pad(
            results['img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['medimg__shape'] = padded_img.shape[1:]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get('gt_seg_map', None) is not None:
            pad_val = self.pad_val.get('seg', 0)

            results['gt_seg_map'] = self._to_pad(
                results['gt_seg_map'],
                shape=results['pad_shape'][1:],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

    @staticmethod
    def _to_pad(
        img: np.ndarray,
        *,
        shape: Optional[Tuple[int, int, int]] = None,
        pad_val: Union[float, List] = 0,
        padding_mode: str = 'constant',
    ) -> np.ndarray:
        """Pad the given 3d image to a certain shape with specified padding
        mode and padding value.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (z, x, y).
                Default: None.
            pad_val (Number | Sequence[Number]): Values to be filled
                in padding areas when padding_mode is 'constant'. Default: 0.
            padding_mode (str): Type of padding. Should be: constant, edge,
                reflect or symmetric. Default: constant.
                - constant: pads with a constant value, this value
                is specified with pad_val.
                - edge: pads with the last value at the edge of the image.
                - reflect: pads with reflection of image without repeating
                the last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result in
                [3, 2, 1, 2, 3, 4, 3, 2].
                - symmetric: pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with 2
                elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3]
        Returns:
            ndarray: The padded image.
        """
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        if padding_mode != 'constant' and pad_val != 0:
            warnings.warn('``pad_val`` is only support in mode=``constant``,'
                          'the pad_val will be ignore in other padding_mode')

        if shape is not None:
            if not (is_tuple_of(shape, int) and len(shape) == 3):
                raise ValueError('Expected padding shape must be a tuple of 3'
                                 f'int element, But receive: {shape}')
            pad_width = []
            for i, sp_i in enumerate(shape):
                if len(img.shape) == 3:
                    width = max(sp_i - img.shape[:][i], 0)
                else:
                    width = max(sp_i - img.shape[1:][i], 0)
                pad_width.append((width // 2, width - (width // 2)))
            if len(img.shape) == 4:
                pad_width = [(0, 0)] + pad_width

        if padding_mode == 'constant':
            img = np.pad(
                img,
                pad_width=pad_width,
                mode='constant',
                constant_values=pad_val)
        else:
            img = np.pad(img, pad_width=pad_width, mode=padding_mode)

        return img

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'padding_mode={self.padding_mode})'
        return

@TRANSFORMS.register_module()
class ZNormalization(BaseTransform):
    """z_normalization.
    # This class is modified from `MONAI.
    # https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/intensity/array.py#L605
    # Copyright (c) MONAI Consortium
    # Licensed under the Apache License, Version 2.0 (the "License")
    Required Keys:
    - img
    Modified Keys:
    - img
    Args:
        mean (float, optional): the mean to subtract by
            Defaults to None.
        std (float, optional): the standard deviation to divide by
            Defaults to None.
        nonzero: whether only normalize non-zero values
            Defaults to False.
        channel_wise (bool): whether perform channel-wise znormalization
            Defaults to False.
    """

    def __init__(self,
                 mean: Optional[Union[float, Iterable[float]]] = None,
                 std: Optional[Union[float, Iterable[float]]] = None,
                 nonzero: bool = False,
                 channel_wise: bool = False) -> None:
        self.mean = mean
        self.std = std
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def _normalize(self, img: np.ndarray, mean=None, std=None):
        if self.nonzero:
            slices = img != 0
        else:
            slices = np.ones_like(img, dtype=bool)

        if not slices.any():
            return img

        _mean = mean if mean is not None else np.mean(img[slices])
        _std = std if std is not None else np.std(img[slices])

        if np.isscalar(_std):
            if _std == 0.0:
                _std = 1.0
        else:
            _std = _std[slices]
            _std[_std == 0.0] = 1.0

        img[slices] = (img[slices] - _mean) / _std
        return img

    def znorm(self, img):
        if self.channel_wise:
            if self.mean is not None and len(self.mean) != len(img):
                err_str = (f'img has {len(img)} channels, '
                           f'but mean has {len(self.mean)}.')
                raise ValueError(err_str)
            if self.std is not None and len(self.std) != len(img):
                err_str = (f'img has {len(img)} channels, '
                           f'but std has {len(self.std)}.')
                raise ValueError(err_str)

            for i, d in enumerate(img):
                img[i] = self._normalize(
                    d,
                    mean=self.mean[i] if self.mean is not None else None,
                    std=self.std[i] if self.std is not None else None,
                )
        else:
            img = self._normalize(img, self.mean, self.std)

        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        img = self.znorm(img)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, '
        repr_str += f'std={self.std}, '
        repr_str += f'channel_wise={self.channel_wise})'
        return

@TRANSFORMS.register_module()
class RandomRotFlip(BaseTransform):
    """Rotate and flip the image & seg or just rotate the image & seg.
    Required Keys:
    - img
    - gt_seg_map
    Modified Keys:
    - img
    - gt_seg_map
    Args:
        rotate_prob (float): The probability of rotate image.
        flip_prob (float): The probability of rotate&flip image.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
    """

    def __init__(self, rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        assert 0 <= rotate_prob <= 1 and 0 <= flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'

    def random_rot_flip(self, results: dict) -> dict:
        k = np.random.randint(0, 4)
        results['img'] = np.rot90(results['img'], k)
        for key in results.get('seg_fields', []):
            results[key] = np.rot90(results[key], k)
        axis = np.random.randint(0, 2)
        results['img'] = np.flip(results['img'], axis=axis).copy()
        for key in results.get('seg_fields', []):
            results[key] = np.flip(results[key], axis=axis).copy()
        return results

    def random_rotate(self, results: dict) -> dict:
        angle = np.random.uniform(min(*self.degree), max(*self.degree))
        results['img'] = mmcv.imrotate(results['img'], angle=angle)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(results[key], angle=angle)
        return results

    def transform(self, results: dict) -> dict:
        """Call function to rotate or rotate & flip image, semantic
        segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Rotated or rotated & flipped results.
        """
        rotate_flag = 0
        if random.random() < self.rotate_prob:
            results = self.random_rotate(results)
            rotate_flag = 1
        if random.random() < self.flip_prob and rotate_flag == 0:
            results = self.random_rot_flip(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_prob={self.rotate_prob}, ' \
                    f'flip_prob={self.flip_prob}, ' \
                    f'degree={self.degree})'
        return

@TRANSFORMS.register_module()
class MedicalRandomFlip(BaseTransform):
    """Reverse the orders of elements in an 3D Medical image & gt_seg_map along
    a given axe.
    Required Keys:
    - img
    - gt_seg_map
    Modified Keys:
    - img
    - gt_seg_map
    Added Keys:
    - flip
    - flip_direction
    - swap_seg_labels (optional)
    Args:
        prob (float | list[float]): The flipping probability. Defaults to None.
        direction (int, list[int]): The flipping direction (Spatial axes along
            which to flip over).
        swap_seg_labels (list, optional): The label pair need to be swapped
            for ground truth, like 'left arm' and 'right arm' need to be
            swapped after horizontal flipping. For example, ``[(1, 5)]``,
            where 1/5 is the label of the left/right arm. Defaults to None.
    """

    def __init__(self,
                 prob: Optional[Union[float, Iterable[float]]] = None,
                 direction: Optional[Union[Sequence[int], int]] = None,
                 swap_seg_labels: Optional[Sequence] = None) -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(
                f'probs must be float or list, but got `{type(prob)}`.')
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        if isinstance(direction, int):
            pass
        elif isinstance(direction, list):
            assert is_list_of(direction, int)
        else:
            raise ValueError(f'direction must be either int or list of int, \
                               but got `{type(direction)}`.')
        self.direction = direction

    def _choose_direction(self) -> int:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      Sequence) and not isinstance(self.direction, int):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, int):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip_seg_map(self, seg_map: dict, direction: int) -> np.ndarray:
        """
        Args:
            seg_map (ndarray): segmentation map, shape (Z, Y, X)
            direction (int): Flip direction. Options are '0' , '1', '2'
        Returns:
            numpy.ndarray: Flipped segmentation map.
        """
        seg_map = np.flip(seg_map, direction)
        if self.swap_seg_labels is not None:
            # to handle datasets with left/right annotations
            # like 'Left-arm' and 'Right-arm' in LIP dataset
            # Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/datasets/tools/cv2_aug_transforms.py # noqa:E501
            # Licensed under MIT license
            temp = seg_map.copy()
            assert isinstance(self.swap_seg_labels, (tuple, list))
            for pair in self.swap_seg_labels:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2, \
                    'swap_seg_labels must be a sequence with pair, but got ' \
                    f'{self.swap_seg_labels}.'
                seg_map[temp == pair[0]] = pair[1]
                seg_map[temp == pair[1]] = pair[0]

        return seg_map

    def _flip(self, results: dict) -> None:
        """Flip images and segmentation map."""
        # flip image
        results['img'] = np.flip(results['img'], results['flip_direction'] + 1)
        results['gt_seg_map'] = self._flip_seg_map(
            results['gt_seg_map'], direction=results['flip_direction'])
        results['swap_seg_labels'] = self.swap_seg_labels

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images and segmentation map."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'img', 'gt_seg_map', 'flip',
            and 'flip_direction' keys are updated in result dict.
        """
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'

        return repr_str

@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, np.random.uniform(
            min(*self.degree), max(*self.degree))

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate, degree = self.generate_degree()
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@TRANSFORMS.register_module()
class RGB2Gray(BaseTransform):
    """Convert RGB image to grayscale image.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def transform(self, results: dict) -> dict:
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str


@TRANSFORMS.register_module()
class AdjustGamma(BaseTransform):
    """Using gamma correction to process the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def transform(self, results: dict) -> dict:
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@TRANSFORMS.register_module()
class SegRescale(BaseTransform):
    """Rescale semantic segmentation maps.

    Required Keys:

    - gt_seg_map

    Modified Keys:

    - gt_seg_map

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def transform(self, results: dict) -> dict:
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class RandomCutOut(BaseTransform):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): cutout probability.
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
        seg_fill_in (int): The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default: None.
    """

    def __init__(self,
                 prob,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None):

        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    @cache_randomness
    def do_cutout(self):
        return np.random.rand() < self.prob

    @cache_randomness
    def generate_patches(self, results):
        cutout = self.do_cutout()

        h, w, _ = results['img'].shape
        if cutout:
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        else:
            n_holes = 0
        x1_lst = []
        y1_lst = []
        index_lst = []
        for _ in range(n_holes):
            x1_lst.append(np.random.randint(0, w))
            y1_lst.append(np.random.randint(0, h))
            index_lst.append(np.random.randint(0, len(self.candidates)))
        return cutout, n_holes, x1_lst, y1_lst, index_lst

    def transform(self, results: dict) -> dict:
        """Call function to drop some regions of image."""
        cutout, n_holes, x1_lst, y1_lst, index_lst = self.generate_patches(
            results)
        if cutout:
            h, w, c = results['img'].shape
            for i in range(n_holes):
                x1 = x1_lst[i]
                y1 = y1_lst[i]
                index = index_lst[i]
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                results['img'][y1:y2, x1:x2, :] = self.fill_in

                if self.seg_fill_in is not None:
                    for key in results.get('seg_fields', []):
                        results[key][y1:y2, x1:x2] = self.seg_fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in}, '
        repr_str += f'seg_fill_in={self.seg_fill_in})'
        return repr_str


@TRANSFORMS.register_module()
class RandomMosaic(BaseTransform):
    """Mosaic augmentation. Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_seg_map
    - mix_results

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    - gt_seg_map

    Args:
        prob (float): mosaic probability.
        img_scale (Sequence[int]): Image size after mosaic pipeline of
            a single image. The size of the output image is four times
            that of a single image. The output image comprises 4 single images.
            Default: (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default: (0.5, 1.5).
        pad_val (int): Pad value. Default: 0.
        seg_pad_val (int): Pad value of segmentation map. Default: 255.
    """

    def __init__(self,
                 prob,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 pad_val=0,
                 seg_pad_val=255):
        assert 0 <= prob and prob <= 1
        assert isinstance(img_scale, tuple)
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @cache_randomness
    def do_mosaic(self):
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        mosaic = self.do_mosaic()
        if mosaic:
            results = self._mosaic_transform_img(results)
            results = self._mosaic_transform_seg(results)
        return results

    def get_indices(self, dataset: MultiImageMixDataset) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    @cache_randomness
    def generate_mosaic_center(self):
        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        return center_x, center_y

    def _mosaic_transform_img(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        self.center_x, self.center_y = self.generate_mosaic_center()
        center_position = (self.center_x, self.center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = copy.deepcopy(results)
            else:
                result_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = result_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape

        return results

    def _mosaic_transform_seg(self, results: dict) -> dict:
        """Mosaic transform function for label annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        for key in results.get('seg_fields', []):
            mosaic_seg = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.seg_pad_val,
                dtype=results[key].dtype)

            # mosaic center x, y
            center_position = (self.center_x, self.center_y)

            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    result_patch = copy.deepcopy(results)
                else:
                    result_patch = copy.deepcopy(results['mix_results'][i - 1])

                gt_seg_i = result_patch[key]
                h_i, w_i = gt_seg_i.shape[:2]
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[0] / h_i,
                                    self.img_scale[1] / w_i)
                gt_seg_i = mmcv.imresize(
                    gt_seg_i,
                    (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                    interpolation='nearest')

                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, gt_seg_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_seg[y1_p:y2_p, x1_p:x2_p] = gt_seg_i[y1_c:y2_c,
                                                            x1_c:x2_c]

            results[key] = mosaic_seg

        return results

    def _mosaic_combine(self, loc: str, center_position_xy: Sequence[float],
                        img_shape_wh: Sequence[int]) -> tuple:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'seg_pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class GenerateEdge(BaseTransform):
    """Generate Edge for CE2P approach.

    Edge will be used to calculate loss of
    `CE2P <https://arxiv.org/abs/1809.05996>`_.

    Modified from https://github.com/liutinglt/CE2P/blob/master/dataset/target_generation.py # noqa:E501

    Required Keys:

        - img_shape
        - gt_seg_map

    Added Keys:
        - gt_edge (np.ndarray, uint8): The edge annotation generated from the
            seg map by extracting border between different semantics.

    Args:
        edge_width (int): The width of edge. Default to 3.
        ignore_index (int): Index that will be ignored. Default to 255.
    """

    def __init__(self, edge_width: int = 3, ignore_index: int = 255) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.ignore_index = ignore_index

    def transform(self, results: Dict) -> Dict:
        """Call function to generate edge from segmentation map.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with edge mask.
        """
        h, w = results['img_shape']
        edge = np.zeros((h, w), dtype=np.uint8)
        seg_map = results['gt_seg_map']

        # down
        edge_down = edge[1:h, :]
        edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
                  & (seg_map[1:h, :] != self.ignore_index) &
                  (seg_map[:h - 1, :] != self.ignore_index)] = 1
        # left
        edge_left = edge[:, :w - 1]
        edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
                  & (seg_map[:, :w - 1] != self.ignore_index) &
                  (seg_map[:, 1:w] != self.ignore_index)] = 1
        # up_left
        edge_upleft = edge[:h - 1, :w - 1]
        edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                    & (seg_map[:h - 1, :w - 1] != self.ignore_index) &
                    (seg_map[1:h, 1:w] != self.ignore_index)] = 1
        # up_right
        edge_upright = edge[:h - 1, 1:w]
        edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                     & (seg_map[:h - 1, 1:w] != self.ignore_index) &
                     (seg_map[1:h, :w - 1] != self.ignore_index)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.edge_width, self.edge_width))
        edge = cv2.dilate(edge, kernel)

        results['gt_edge'] = edge
        results['edge_width'] = self.edge_width

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'edge_width={self.edge_width}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str
