# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import random
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage

from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint
from training.dataset.transforms import RandomAffine, get_size_with_aspect_ratio


def hflip_with_png(datapoint, index):
    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)
        if hasattr(obj, 'point') and obj.point is not None:
            obj.point = torch.tensor([datapoint.frames[index].data.size[0] - obj.point[0], obj.point[1]])
        if hasattr(obj, 'box') and obj.box is not None:
            obj.box = torch.tensor([datapoint.frames[index].data.size[0] - obj.box[2], obj.box[1], datapoint.frames[index].data.size[0] - obj.box[0], obj.box[3]])
        if hasattr(obj, 'png') and obj.png is not None:
            obj.png = F.hflip(obj.png)
    return datapoint


def resize_with_png(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if hasattr(obj, 'point'):   # PointObject
                segment = F.resize(obj.segment[None, None], size, interpolation=InterpolationMode.NEAREST).squeeze()
                if obj.segment.max() > 0 and segment.max() == 0 :
                    # The object is not visible in the resized segment
                    original_indices = torch.nonzero(obj.segment, as_tuple=True)    # check ok if multiple points
                    new_indices = (original_indices[0] * size[0] // obj.segment.shape[0], original_indices[1] * size[1] // obj.segment.shape[1])
                    segment[new_indices[0], new_indices[1]] = 1
                    obj.segment = segment
                else:
                    obj.segment = segment
            else:
                obj.segment = F.resize(obj.segment[None, None], size, interpolation=InterpolationMode.NEAREST).squeeze()

        if hasattr(obj, 'png') and obj.png is not None:
            obj.png = F.resize(obj.png[None, None], size, interpolation=InterpolationMode.NEAREST).squeeze()

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def pad_with_png(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
        if hasattr(obj, 'png') and obj.png is not None:
            if v2:
                if len(padding) == 2:
                    obj.png = Fv2.pad(obj.png, (0, 0, padding[0], padding[1]))
                else:
                    obj.png = Fv2.pad(obj.png, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.png = F.pad(obj.png, (0, 0, padding[0], padding[1]))
                else:
                    obj.png = F.pad(obj.png, tuple(padding))
    return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = hflip_with_png(datapoint, i)
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = hflip_with_png(datapoint, i)
        return datapoint


class RandomResizeAPI_mask_with_NEAREST_and_png:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.frames)):
                datapoint = resize_with_png(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = resize_with_png(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class RandomAffine_with_points_check(RandomAffine):
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):  # Apply safe affine transformation to the first frame, make sure the box and point is full and not cropped.
        super(RandomAffine_with_points_check, self).__init__(
            degrees=degrees,
            consistent_transform=consistent_transform,
            scale=scale,
            translate=translate,
            shear=shear,
            image_mean=image_mean,
            log_warning=log_warning,
            num_tentatives=num_tentatives,
            image_interpolation=image_interpolation,
        )

    def transform_datapoint(self, datapoint: VideoDatapoint):
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            this_boxes_xyxy = [
                obj.box if (hasattr(obj, 'box') and obj.box is not None) else None
                for obj in img.objects
            ]
            this_points_xy = [
                obj.box if (hasattr(obj, 'point') and obj.point is not None) else None
                for obj in img.objects
            ]
            this_pngs = [
                obj.png.unsqueeze(0) if (hasattr(obj, 'png') and obj.png is not None) else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            transformed_boxes_xyxy, transformed_points_xy, transformed_pngs = [], [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_masks.append(transformed_mask.squeeze())

                if this_boxes_xyxy[i] is None:
                    transformed_boxes_xyxy.append(None)
                    # # Dummy bbox for a dummy target
                    # transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    points = []
                    points.append(torch.tensor([this_boxes_xyxy[i][0], this_boxes_xyxy[i][1]]))   # top-left
                    points.append(torch.tensor([this_boxes_xyxy[i][2], this_boxes_xyxy[i][1]]))   # top-right
                    points.append(torch.tensor([this_boxes_xyxy[i][0], this_boxes_xyxy[i][3]]))   # bottom-left
                    points.append(torch.tensor([this_boxes_xyxy[i][2], this_boxes_xyxy[i][3]]))   # bottom-right

                    transformed_points = RandomAffine_with_points_check._transform_points(points, img_size, affine_params)
                    if img_idx == 0 and not RandomAffine_with_points_check._is_points_in_image(transformed_points, img_size):
                        return None

                    x_min, _ = torch.min(transformed_points[:, 0], dim=0)
                    y_min, _ = torch.min(transformed_points[:, 1], dim=0)
                    x_max, _ = torch.max(transformed_points[:, 0], dim=0)
                    y_max, _ = torch.max(transformed_points[:, 1], dim=0)
                    transformed_box_xyxy = torch.tensor([x_min, y_min, x_max, y_max])
                    transformed_boxes_xyxy.append(transformed_box_xyxy)
                    
                if this_points_xy[i] is None:
                    transformed_points_xy.append(None)
                    # # Dummy bbox for a dummy target
                    # transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    points = []
                    points.append(torch.tensor([this_points_xy[i][0], this_points_xy[i][1]]))

                    transformed_points = RandomAffine_with_points_check._transform_points(points, img_size, affine_params)
                    if img_idx == 0 and not RandomAffine_with_points_check._is_points_in_image(transformed_points, img_size):
                        return None
                    
                    transformed_points_xy.append(transformed_points[0])

                if this_pngs[i] is None:
                    transformed_pngs.append(None)
                    # # Dummy bbox for a dummy target
                    # transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_png = F.affine(
                        this_pngs[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    # if img_idx == 0 and transformed_png.max() == 0:
                    #     # We are dealing with a video and the object is not visible in the first frame
                    #     # Return the datapoint without transformation
                    #     return None
                    transformed_pngs.append(transformed_png.squeeze())

            for i in range(len(img.objects)):
                img.objects[i].segment = transformed_masks[i]
                if hasattr(img.objects[i], 'box'):
                    img.objects[i].box = transformed_boxes_xyxy[i]
                if hasattr(img.objects[i], 'point'):
                    img.objects[i].point = transformed_points_xy[i]
                if hasattr(img.objects[i], 'png'):
                    img.objects[i].png = transformed_pngs[i]

            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint
    
    @staticmethod
    def _transform_points(points:list[torch.tensor], img_size, affine_params):
        # Helper function to transform the box
        center = [img_size[0] * 0.5, img_size[1] * 0.5] # https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.functional.affine
        affine_matrix = F._get_inverse_affine_matrix(center, *affine_params, inverted=False)

        points = torch.stack(points) # (N, 2)
        
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_ones = torch.cat([points, ones], dim=1)  # [N, 3]
        
        M = torch.tensor(affine_matrix).view(2, 3)

        # Apply the affine transformation
        transformed_points = (M @ points_ones.T).T
        return transformed_points   # [N, 2]
    
    @staticmethod
    def _is_points_in_image(transformed_points, img_size):
        x_coords = transformed_points[:, 0]
        y_coords = transformed_points[:, 1]
        within_x = x_coords.min() >= 0 and x_coords.max() <= img_size[0]
        within_y = y_coords.min() >= 0 and y_coords.max() <= img_size[1]
        return within_x and within_y