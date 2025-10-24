# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


from training.utils.data_utils import BatchedVideoMetaData

@tensorclass
class BatchedVideoMetaDataWithBoxesPoints(BatchedVideoMetaData):
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
        task: A string indicating the type of object in the batch, including ['mask', 'box', 'point']
    """
    task: str = "mask"



from training.utils.data_utils import BatchedVideoDatapoint

@tensorclass
class BatchedVideoDatapointWithBoxesPoints(BatchedVideoDatapoint):
    masks: Union[torch.BoolTensor, torch.FloatTensor]   # masks: torch.BoolTensor
    metadata: BatchedVideoMetaDataWithBoxesPoints       # metadata: BatchedVideoMetaData

    points: Optional[torch.IntTensor]
    boxes: Optional[torch.FloatTensor]
    visibles: Optional[torch.BoolTensor]



from training.utils.data_utils import Object

@dataclass
class PointObject(Object):
    point: Union[torch.Tensor, dict] = None     # center point (for point tracking)
    visible: Optional[bool] = None              # Whether the object is visible in the frame

@dataclass
class BoxObject(Object):
    box: Union[torch.Tensor, dict] = None       # bounding box (for box tracking) xyxy format
    visible: Optional[bool] = None              # Whether the object is visible in the frame
    png: Union[torch.Tensor, dict] = None       # temporary variable for extracted png in dataloader, not return to model.



from training.utils.data_utils import VideoDatapoint

def collate_fn_plus(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapointWithBoxesPoints:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    if isinstance(batch[0].frames[0].objects[0], BoxObject):
        task = "box"
    elif isinstance(batch[0].frames[0].objects[0], PointObject):
        task = "point"
    else:
        task = "mask"

    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_point = [[] for _ in range(T)]
    step_t_boxes = [[] for _ in range(T)]
    step_t_visibles = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )

                # mask data
                # step_t_masks[t].append(obj.segment.to(torch.bool))
                if task == "mask":
                    step_t_masks[t].append(obj.segment.to(torch.bool))
                elif task == "point":
                    step_t_masks[t].append(obj.segment)
                elif task == "box":
                    step_t_masks[t].append(obj.segment.to(torch.bool))
                else:
                    raise ValueError(f"Invalid task: {task}")
                
                # point data
                if task == "point":
                    step_t_point[t].append(obj.point)
                
                # box data
                if task == "box":
                    step_t_boxes[t].append(obj.box)

                # visible data
                if task == "mask":
                    pass
                elif task == "point":
                    step_t_visibles[t].append(obj.visible)
                elif task == "box":
                    step_t_visibles[t].append(obj.visible)
                else:
                    raise ValueError(f"Invalid task: {task}")

                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )

    # mask data
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    
    # point data
    if task == "point":
        points = torch.stack([torch.stack(point, dim=0) for point in step_t_point], dim=0)
    else:
        points = None
    
    # box data
    if task == "box":
        boxes = torch.stack([torch.stack(box, dim=0) for box in step_t_boxes], dim=0)
    else:
        boxes = None

    # visible data
    if task == "box" or task == "point":
        visibles = torch.stack([torch.tensor(visible) for visible in step_t_visibles], dim=0)
    elif task == "mask":
        visibles = torch.stack([torch.stack([torch.any(mask) for mask in masks]) for masks in step_t_masks], dim=0)
    else:
        raise ValueError(f"Invalid task: {task}")

    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapointWithBoxesPoints(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        points=points,
        boxes=boxes,
        visibles=visibles,
        metadata=BatchedVideoMetaDataWithBoxesPoints(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
            task=task,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )