# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code borrowed from TLC - https://www.internalfb.com/code/fbsource/fbcode/pytorch/tlc/torchtlc/loggers/tensorboard.py
import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union, List

from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from training.utils.train_utils import get_machine_local_and_dist_rank, makedir

Scalar = Union[Tensor, ndarray, int, float]

from training.utils.logger import Logger, TensorBoardLogger
from training.utils_plus.visualization import TensorboardVisualizer

from PIL import Image
import io
import numpy as np
import torchvision.transforms as T

def _tensor_to_png(tensor, savepath):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor * 255).astype(np.uint8)
    img = Image.fromarray(tensor)
    img.save(savepath)

def _maskTensor_to_png(mask_tensor, savepath):
    mask_tensor = mask_tensor.detach().cpu().numpy()
    mask_tensor = (mask_tensor * 255).astype(np.uint8)
    img = Image.fromarray(mask_tensor)
    img.save(savepath)

def make_tensorboard_logger_plus(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger_plus(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


class TensorBoardLogger_plus(TensorBoardLogger):
    def log_image(self, tag: str, img_tensor: Any, step: int) -> None:
        """Add image data to TensorBoard.

        Args:
            tag (string): tag name used to group images
            img_tensor (Tensor or numpy array): image data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_image(tag, img_tensor, global_step=step)


class Logger_plus(Logger):
    def __init__(self, logging_conf):
        super(Logger_plus, self).__init__(logging_conf)
        # set the mean and std for denormalization
        self.data_visulizer = TensorboardVisualizer()

    def log_image(self, tag: str, img_tensor: Any, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_image(tag, img_tensor, step)
    
    def log_visualize_multi_task_io(
            self,
            task: str, step: int,
            phase: str,
            input_imgs: Tensor, 
            gt_segments: Tensor, gt_boxes: Tensor, gt_points: Tensor,
            output_segments: Tensor, output_boxes: Tensor, output_points: Tensor,
            prompt_points: List[Optional[Tensor]], prompt_masks: List[Optional[Tensor]],
            obj_to_frame_idx: Optional[Tensor],
        ):
        """
        @param task: str, the task name, [mask, box, point]
        @param step: int, the step number
        @param input_imgs: Tensor, [T, B, 3, H, W], N is the number of objects.

        @param gt_segments: Tensor, [T, B*N, H, W], the ground truth segments, where N is the number of objects.
        @param gt_boxes: Tensor, [T, B*N, 4], the ground truth boxes. XYXY format. N = 1.
        @param gt_points: Tensor, [T, B*N, 2], the ground truth points. N = 1.

        @param output_segments: Tensor, [T, B*N, 1, H, W], the output segments.
        @param output_boxes: Tensor, [T, B*N, 1, 4], the output boxes. XYXY format. N = 1.
        @param output_points: Tensor, [T, B*N, 1, 2], the output points. XY format.

        @param prompt_points: List of dict, Length is T, each dict is {'point_coords': Tensor [B, P, 2], 'point_labels': Tensor [B, P]}, where P is the number of points.
        @param prompt_masks: List of Tensor, Length is T, each tensor is [B*N, 1, H, W]

        """
        if task == 'mask' or task == 'point':
            output_segments = output_segments.sigmoid()
        # clone the input to avoid the inplace operation
        input_imgs = input_imgs.clone().detach().cpu()
        gt_segments = gt_segments.clone().detach().cpu()
        gt_boxes = gt_boxes.clone().detach().cpu() if gt_boxes is not None else None
        gt_points = gt_points.clone().detach().cpu() if gt_points is not None else None
        output_segments = output_segments.clone().detach().cpu()
        output_boxes = output_boxes.clone().detach().cpu() if output_boxes is not None else None
        output_points = output_points.clone().detach().cpu() if output_points is not None else None
        prompt_points = [self._clone_dict(prompt_point) if prompt_point is not None else None for prompt_point in prompt_points]
        prompt_masks = [prompt_mask.clone().detach().cpu() if prompt_mask is not None else None for prompt_mask in prompt_masks]

        # Filter batch, keep 1 sample
        N = (obj_to_frame_idx[0][:,1]==0).sum().item()
        # if N > 1:
        #     print("Multi Objects")

        select_input_imgs = input_imgs[:,:1].squeeze(1) # [T, 3, H, W]
        select_gt_segments = gt_segments[:,:N] # [T, N, H, W]
        select_gt_boxes = gt_boxes[:,:N] if gt_boxes is not None else None # [T, N, 4]
        select_gt_points = gt_points[:,:N] if gt_points is not None else None # [T, N, 2]
        select_output_segments = output_segments[:, :N].squeeze(2) # [T, N, H, W]
        select_output_boxes = output_boxes[:, :N].squeeze(2) if output_boxes is not None else None # [T, N, 4]
        select_output_points = output_points[:, :N].squeeze(2) if output_points is not None else None # [T, N, 2]
        select_prompt_points = [self._select_dict(prompt_point, N) if prompt_point is not None else None for prompt_point in prompt_points] # T * [{'point_coords': [1, P, 2], 'point_labels': [1, P]}]
        select_prompt_masks = [prompt_mask[:N] if prompt_mask is not None else None for prompt_mask in prompt_masks] # T * [N, 1, H, W]/None
        

        logged_img = self.data_visulizer.log_visualize_multi_task_io(
            task=task,
            input_imgs=select_input_imgs,

            gt_segments=select_gt_segments,
            gt_boxes=select_gt_boxes,
            gt_points=select_gt_points,

            output_segments=select_output_segments,
            output_boxes=select_output_boxes,
            output_points=select_output_points,

            prompt_points=select_prompt_points,
            prompt_masks=select_prompt_masks
        )

        
        # 3. log the images
        self.log_image(f"Visualization/{phase}/{task}", logged_img, step)
    
    def _clone_dict(self, d):
        return {k: v.clone().detach().cpu() if v is not None else None for k, v in d.items()}
    
    def _select_dict(self, d, topn):
        return {k: v[:topn] if v is not None else None for k, v in d.items()}
    