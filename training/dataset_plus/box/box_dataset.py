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
from training.utils_plus.data_utils import BoxObject

from training.dataset.vos_dataset import MAX_RETRIES

from training.dataset.vos_dataset import VOSDataset, load_images, tensor_2_PIL

from training.dataset_plus.box.box_raw_dataset import BoxRawDataset
from training.dataset_plus.box.utils import box_to_mask
from torchvision.ops import masks_to_boxes


class BoxDataset(VOSDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: BoxRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
        samples_per_epoch: int = None,
    ):
        super().__init__(transforms, training, video_dataset, sampler, multiplier, always_target, target_segments_available)
        self.samples_per_epoch = samples_per_epoch

    def _get_datapoint(self, idx):

        # get box and image, generate mask from box, transform the image and mask
        data_point = super()._get_datapoint(idx)

        # transforms use the generated square-mask instead of box, so we must generate box from transformed square-mask
        data_point = self._generate_boxes_from_transformed_masks(data_point)

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
                raise AttributeError("JSONSegmentLoader not for BoxRawDataset")
            else:
                # segments = segment_loader.load(frame.frame_idx)
                boxes, visibles, pngs = segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if visibles[obj_id]:
                    assert (
                        torch.all(boxes[obj_id] >= 0)
                    ), f"None targets are not supported, Box {boxes[obj_id]} is invalid"
                    box = boxes[obj_id]
                    # clip the box to the image size
                    # if box[0] > w or box[1] > h or box[2] > w or box[3] > h:
                    #     logging.warning(
                    #         f"Box {box} is outside of the image size {w}x{h}, clip it to the image size, frame_idx: {frame_idx}, video_name: {video.video_name}"
                    #     )
                    box[0] = torch.clip(box[0], 0, w)   # x_min
                    box[1] = torch.clip(box[1], 0, h)   # y_min
                    box[2] = torch.clip(box[2], 0, w)   # x_max
                    box[3] = torch.clip(box[3], 0, h)   # y_max
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = box_to_mask(box, h, w, target_visible=torch.tensor(visibles[obj_id], device=box.device)).squeeze(0).to(torch.uint8)
                    visible = True
                    png = pngs[obj_id].to(torch.uint8) if pngs[obj_id] is not None else None
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    box = torch.tensor([0, 0, 0, 0])
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                    visible = False
                    png = torch.zeros(h, w, dtype=torch.uint8) if pngs[obj_id] is not None else None

                images[frame_idx].objects.append(
                    BoxObject(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                        box=box, 
                        visible=visible,
                        png=png,
                    )
                )
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def _check_first_frame_objs_all_exist(self, datapoint):
        return all([obj.segment.sum() >= 4 for obj in datapoint.frames[0].objects])

    def _generate_boxes_from_transformed_masks(self, data_point):
        """
        Generate the box from the transformed mask.
        If the mask is empty, generate a zero mask or remove the object.
        """
        for frame in data_point.frames:
            # if isinstance(frame.data, torch.Tensor):
            #     h, w = frame.data.shape[-2:]
            # elif isinstance(frame.data, PIL.Image.Image):
            #     w, h = frame.data.size
            # else:
            #     raise ValueError("frame.data should be a tensor or PIL image")
            assert isinstance(frame.data, torch.Tensor)

            to_remove = []
            for obj in frame.objects:
                if obj.segment.sum() >= 4:
                    obj.box = masks_to_boxes(obj.segment.unsqueeze(0)).squeeze(0)
                    obj.visible = True
                    if obj.png is not None:
                        obj.segment = obj.png
                    obj.png = None
                else:
                    if self.always_target:
                        obj.box = torch.tensor([0, 0, 0, 0])    # empty box
                        obj.visible = False
                        obj.segment = torch.zeros_like(obj.segment)
                        obj.png = None
                    else:
                        to_remove.append(obj)
            for obj in to_remove:
                frame.objects.remove(obj)

        return data_point

    def __getitem__(self, idx):
        if self.samples_per_epoch is not None:
            if self.samples_per_epoch < len(self.video_dataset):
                idx += random.randint(0, len(self.video_dataset) - 1)
            idx = idx % len(self.video_dataset)
        return self._get_datapoint(idx)

    def __len__(self):
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        else:
            return len(self.video_dataset)