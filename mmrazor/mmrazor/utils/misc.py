# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import numpy as np
import cv2
from typing import List, Optional, Union, Tuple

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def show_featmaps(feats,
                  show=False,
                  overlaid=True,
                  resize_shape=(512, 512),
                  is_image=False,
                  channel_reduction='squeeze_mean',
                  topk: int = 20,
                  arrangement: Tuple[int, int] = (4, 5)):
    from mmengine.visualization import Visualizer
    visualizer = Visualizer.get_current_instance()
    import matplotlib.pyplot as plt
    if is_image:
        ori_img = feats
        ori_img = feats.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        # ori_img = np.uint8(ori_img / ori_img.max()) * 255
        norm_img = np.zeros(ori_img.shape)
        norm_img = cv2.normalize(ori_img, norm_img, 0, 255, cv2.NORM_MINMAX)
        ori_img = np.asarray(norm_img, dtype=np.uint8)
        visualizer.ori_img = ori_img
        warnings.warn("ori_img of visualizer is None! Make sure that feats is ori_img.")
        if show:
            plt.imshow(visualizer.ori_img)
            plt.show()
        return ori_img
    if overlaid:
        if visualizer.ori_img is None:
            visualizer.ori_img = feats.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            warnings.warn("ori_img of visualizer is None! Make sure that feats is ori_img.")
            plt.imshow(visualizer.ori_img)
            plt.show()
            return
        ori_img = visualizer.ori_img
    else:
        ori_img = None

    featmaps = visualizer.draw_featmap(feats,
                                       ori_img,
                                       resize_shape=resize_shape,
                                       channel_reduction=channel_reduction,
                                       topk=topk,
                                       arrangement=arrangement)
    if show:
        plt.imshow(featmaps)
        plt.show()
    return featmaps


def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension. Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.

    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path
