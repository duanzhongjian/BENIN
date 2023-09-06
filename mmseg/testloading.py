import mmcv

from tests.test_datasets.test_loading import TestLoading
# from tests.test_datasets.test_dataset import test_decathlon
from mmseg.datasets.transforms import *  # noqa
import os.path as osp
import copy
from mmengine.structures import BaseDataElement

from mmseg.structures import SegDataSample
import numpy as np
from mmseg.registry import TRANSFORMS
import torch
from mmengine.utils import scandir
from mmseg.models.decode_heads import SegformerHead
from mmengine.utils import ProgressBar
def main():
    path = './data/LITS'
    split_names = ['training', 'validation']
    min = 10000
    for dir in ['images']:
        for split in split_names:
            folder = osp.join(path, dir, split)
            for img_dir in scandir(folder, recursive=True):
                results = dict(
                    img_path=osp.join(path, dir, split, img_dir),
                )
                transform = LoadBiomedicalImageFromFile()
                # transform2 = LoadAnnotations()
                results = transform(copy.deepcopy(results))
                if results['img'].shape[0] < min:
                    min = results['img'].shape[1]
    print(min)
def test_segformer_head():

    SegformerHead(
        in_channels=(1, 2, 3), in_index=(0, 1), channels=5, num_classes=3)

    H, W, C = (64, 64, 128)
    in_channels = (32, 64, 160, 256)
    shapes = [(H // 2**(i + 2), W // 2**(i + 2))
              for i in range(len(in_channels))]
    model = SegformerHead(
        in_channels=in_channels,
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=19)


    inputs = [torch.randn((1, in_channel, *shape))
              for in_channel, shape in zip(in_channels, shapes)][:3]
    temp = model(inputs)

    # Normal Input
    # ((1, 32, 16, 16), (1, 64, 8, 8), (1, 160, 4, 4), (1, 256, 2, 2)
    inputs = [
        torch.randn((1, in_channel, *shape))
        for in_channel, shape in zip(in_channels, shapes)
    ]
    temp = model(inputs)

    assert temp.shape == (1, 19, H // 4, W // 4)

def test_transform3d():
    results = dict(
        img_path="tests/data/biomedical.nii.gz",
        # img_path = "/root/autodl-nas/datasets/volume-0.nii",

        seg_map_path ='tests/data/biomedical_ann.nii.gz',
        # seg_map_path = 'tests/data/color.jpg',
        # reduce_zero_label=True, seg_fields=[]
    )
    transform1 = LoadBiomedicalImageFromFile()
    transform2 = LoadBiomedicalAnnotation()
    # transform2 = LoadAnnotations()
    results = transform1(copy.deepcopy(results))
    results = transform2(copy.deepcopy(results))
    results['seg_fields'] = ['gt_semantic_seg']
    results['pad_shape'] = results['ori_shape']
    results['scale_factor'] = 1.0

    transform = PackMedicalInputs()
    # transform = PackSegInputs()
    results = transform(copy.deepcopy(results))
def test_transform2d():
    from mmseg.datasets.transforms.loading import LoadImageFromFile
    results = dict(
        img_path = "tests/data/color.jpg",
        seg_map_path = 'tests/data/seg.png',
        reduce_zero_label=True, seg_fields=[]
    )
    transform1 = LoadImageFromFile()
    # transform2 = LoadBiomedicalAnnotation()
    transform2 = LoadAnnotations()
    results = transform1(copy.deepcopy(results))
    results = transform2(copy.deepcopy(results))
    results['seg_fields'] = ['gt_semantic_seg']
    results['pad_shape'] = results['ori_shape']
    results['scale_factor'] = 1.0

    transform = PackSegInputs()
    results = transform(copy.deepcopy(results))
def testloading():
    testloading = TestLoading()
    testloading.data_prefix = 'data'
    testloading.test_load_biomedical_img()
    testloading.test_load_biomedical_annotation()

def test_random_crop():
    # test assertion for invalid random crop
    results = dict(
        img_path="/root/autodl-nas/datasets/biomedical.nii.gz",
        seg_map_path='/root/autodl-nas/datasets/biomedical_ann.nii.gz')
    transform1 = LoadBiomedicalImageFromFile()
    transform2 = LoadBiomedicalAnnotation()
    results = transform1(copy.deepcopy(results))
    results = transform2(copy.deepcopy(results))
    results['seg_fields'] = ['gt_semantic_seg']
    results['pad_shape'] = results['ori_shape']
    results['scale_factor'] = 1.0

    for mode in [None, 'nnUNet']:
        d, h, w = results['img_shape']
        transform = dict(
            type='BioPatchCrop',
            crop_size=(d - 20, h - 20, w - 20),
            crop_mode=mode)
        transform = TRANSFORMS.build(transform)

        results = transform(results)
        assert results['img'].shape[1:] == (d - 20, h - 20, w - 20)
        assert results['img_shape'] == (d - 20, h - 20, w - 20)
        assert results['gt_seg_map'].shape == (d - 20, h - 20, w - 20)

def testdecathlon():
    testdecathlon = test_decathlon()

def testresize():
    # Test `Resize`, `RandomResize` and `RandomChoiceResize` from
    # MMCV transform. Noted: `RandomResize` has args `scales` but
    # `Resize` and `RandomResize` has args `scale`.
    transform = dict(type='Resize', scale=(75, 500, 500), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)

    results = dict(
        img_path=osp.join("/root/autodl-nas/datasets/biomedical.nii.gz"))
    transform = LoadBiomedicalImageFromFile()

    results = transform(copy.deepcopy(results))
    # Set initial values for default meta_keys
    results['pad_shape'] = results['img_shape']
    results['scale_factor'] = 1.0

    resized_results = resize_module(results.copy())
    # img_shape = results['img'].shape[:2] in ``MMCV resize`` function
    # so right now it is (750, 1333) rather than (750, 1333, 3)
    assert resized_results['img_shape'] == (750, 1333)

    # test keep_ratio=False
    transform = dict(
        type='RandomResize',
        scale=(1280, 800),
        ratio_range=(1.0, 1.0),
        resize_type='Resize',
        keep_ratio=False)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (800, 1280)

    # test `RandomChoiceResize`, which in older mmsegmentation
    # `Resize` is multiscale_mode='range'
    transform = dict(type='RandomResize', scale=[(1333, 400), (1333, 1200)])
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333
    assert min(resized_results['img_shape'][:2]) >= 400
    assert min(resized_results['img_shape'][:2]) <= 1200

    # test RandomChoiceResize, which in older mmsegmentation
    # `Resize` is multiscale_mode='value'
    transform = dict(
        type='RandomChoiceResize',
        scales=[(1333, 800), (1333, 400)],
        resize_type='Resize',
        keep_ratio=False)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] in [(800, 1333), (400, 1333)]

    transform = dict(type='Resize', scale_factor=(0.9, 1.1), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333 * 1.1

    # test scale=None and scale_factor is tuple.
    # img shape: (288, 512, 3)
    transform = dict(
        type='Resize', scale=None, scale_factor=(0.5, 2.0), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert int(288 * 0.5) <= resized_results['img_shape'][0] <= 288 * 2.0
    assert int(512 * 0.5) <= resized_results['img_shape'][1] <= 512 * 2.0

    # test minimum resized image shape is 640
    transform = dict(type='Resize', scale=(2560, 640), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (640, 1138)

    # test minimum resized image shape is 640 when img_scale=(512, 640)
    # where should define `scale_factor` in MMCV new ``Resize`` function.
    min_size_ratio = max(640 / img.shape[0], 640 / img.shape[1])
    transform = dict(
        type='Resize', scale_factor=min_size_ratio, keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (640, 1138)

    # test h > w
    img = np.random.randn(512, 288, 3)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    min_size_ratio = max(640 / img.shape[0], 640 / img.shape[1])
    transform = dict(
        type='Resize',
        scale=(2560, 640),
        scale_factor=min_size_ratio,
        keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (1138, 640)

def testpixeldata3d():
    import numpy as np
    from mmseg.structures.pixel_data3d import PixelData3D
    import torch
    metainfo = dict(img_id=np.random.randint(0, 100),
                    img_shape=(1, np.random.randint(400, 600), np.random.randint(400, 600), np.random.randint(400, 600)))
    image = np.random.randint(0, 255, (1, 4, 20, 40))
    featmap = torch.randint(0, 255, (1, 4, 20, 40))
    pixel_data = PixelData3D(metainfo=metainfo, image=image, featmap=featmap)
    print(pixel_data)

def frames2video(
                 fps: float = 30,
                 fourcc: str = 'DIVX') -> None:
    import os
    import cv2
    import mmcv
    from mmseg.engine.hooks.visualization_hook import arr_to_img
    work_dir = './work_dirs'
    results = dict(
        img_path="./data/biomedical.nii.gz",
    )
    transform = LoadBiomedicalImageFromFile()
    results = transform(copy.deepcopy(results))
    imgs = np.rot90(arr_to_img(results['img'][0]))
    height, width = imgs.shape[-2:]
    list = []
    for i in range(imgs.shape[0]):
        list.append(imgs[i].reshape(1, height, width))
    video_file = os.path.join(work_dir, '4.avi')

    resolution = (width, height)

    vwriter = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*fourcc), fps,
                              resolution)
    for img in list:
        img = mmcv.imread(img, backend='cv2')
        vwriter.write(img)

    vwriter.release()
if __name__ == '__main__':
    frames2video()
