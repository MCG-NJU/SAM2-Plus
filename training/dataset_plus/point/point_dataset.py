# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, VideoDatapoint
from training.utils_plus.data_utils import PointObject

from training.dataset.vos_dataset import MAX_RETRIES

from training.dataset.vos_dataset import VOSDataset, load_images, tensor_2_PIL

from training.dataset_plus.point.point_raw_dataset import PointRawDataset
from training.dataset_plus.point.utils import generate_single_point_heatmap, calculate_center_point_from_heatmap, generate_gaussian
import bisect


class PointDataset(VOSDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: PointRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,

        radius_list=50,
        sigma_list=16,
        epoch_list=0,
    ):
        super().__init__(transforms, training, video_dataset, sampler, multiplier, always_target, target_segments_available)

        if isinstance(radius_list, int):
            radius_list = [radius_list]
        if isinstance(sigma_list, int):
            sigma_list = [sigma_list]
        if isinstance(epoch_list, int):
            epoch_list = [epoch_list]
        epoch_list = sorted(epoch_list)
        assert len(radius_list) == len(sigma_list), "radius_list and sigma_list must have the same length"
        assert len(radius_list) == len(epoch_list), "radius_list and epoch_list must have the same length"
        
        self.radius_list = radius_list
        self.sigma_list = sigma_list
        self.epoch_list = epoch_list

        self.radius = self.radius_list[0]
        self.sigma = self.sigma_list[0]

    def set_epoch(self, epoch):
        if hasattr(super(), "set_epoch"):
            super().set_epoch(epoch)

        idx = bisect.bisect_right(self.epoch_list, epoch) - 1
        assert idx >= 0 and idx < len(self.radius_list), f"epoch {epoch} out of range, must be in [0, {self.epoch_list[-1]}]"
        self.radius = self.radius_list[idx]
        self.sigma = self.sigma_list[idx]

    def _get_datapoint(self, idx):

        # get points, init_mask, visible and image, transform the image and init_mask
        data_point = super()._get_datapoint(idx)

        # transforms use the generated single-point-mask instead of point, so we must generate point from transformed single-point-mask and re-generate the guassian-mask from new point
        data_point = self._generate_points_from_transformed_masks_and_regenerate_mask(data_point)

        return data_point

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )
            # We load the gt segments associated with the current frame
            if isinstance(segment_loader, JSONSegmentLoader):
                # segments = segment_loader.load(
                #     frame.frame_idx, obj_ids=sampled_object_ids
                # )
                raise AttributeError("JSONSegmentLoader not for PointDataset")
            else:
                # segments = segment_loader.load(frame.frame_idx)
                points, visibles = segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if visibles[obj_id]:
                    assert (
                        torch.all(points[obj_id] >= 0)
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = generate_single_point_heatmap(points[obj_id], h=h, w=w).to(torch.uint8)
                    point = points[obj_id]
                    visible = True
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                    point = torch.tensor([0, 0])
                    visible = False

                images[frame_idx].objects.append(
                    PointObject(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                        point=point,
                        visible=visible
                    )
                )
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def _check_first_frame_objs_all_exist(self, datapoint):
        return all([obj.segment.sum() >= 1 for obj in datapoint.frames[0].objects])

    def _generate_points_from_transformed_masks_and_regenerate_mask(self, data_point):
        """
        Generate a gaussian mask for each point object.
        If point object is empty, generate a zero mask or remove the object.
        """
        for frame in data_point.frames:
            h, w = frame.data.shape[-2:]
            assert isinstance(frame.data, torch.Tensor)

            to_remove = []
            for obj in frame.objects:
                if obj.segment.sum() >= 1:
                    obj.point = calculate_center_point_from_heatmap(obj.segment)    # calculate the center point of the segment
                    obj.segment = torch.from_numpy(generate_gaussian(obj.point.numpy(), img_size=(h, w), radius=self.radius, sigma=self.sigma))
                    obj.visible = True
                else:
                    if self.always_target:
                        obj.segment = torch.zeros_like(obj.segment)
                        obj.point = torch.tensor([0, 0])  # empty point
                        obj.visible = False
                    else:
                        to_remove.append(obj)
            for obj in to_remove:
                frame.objects.remove(obj)

        return data_point