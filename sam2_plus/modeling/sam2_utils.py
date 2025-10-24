# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box

from sam2.modeling.sam2_utils import sample_one_point_from_error_center


"""
This func is adapted from `sam2.modeling.sam2_utils.sample_box_points` to prepare noise box prompt for the model.
1) given HW instead of get it from the mask 
2) input boxes, so box_coords = mask_to_box(masks)  -->  box_coords = boxes
"""
def sample_noised_box_points(
    H: int, W: int,
    boxes: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - H, W: height and width of the image, dtype=int
    - boxes: [B, 1, 4], GT boxes, dtype=torch.Tensor, xyxy format, not normalized.
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = boxes.device
    box_coords = boxes  # box_coords = mask_to_box(masks)
    B = box_coords.shape[0] # B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors_probabilistic_mask(gt_masks, pred_masks, num_pt=1, thread=0.98):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.
    NOTE: Only gaussian mask is supported.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, torch.float32
    - pred_masks: [B, 1, H_im, W_im] masks, torch.float32 or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    assert gt_masks.dtype == torch.float32 
    assert gt_masks.dtype == torch.float32 

    gt_masks_prob = gt_masks.clone()
    gt_masks = gt_masks > thread
    if not torch.any(gt_masks):
        Warning(f"Not any positive point found in the GroundTruth mask with threshold {thread}")

    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
        pred_masks_prob = torch.zeros_like(gt_masks, dtype=torch.float32)
    else:
        pred_masks_prob = pred_masks.clone()
        pred_masks = pred_masks > thread
        if not torch.any(pred_masks):
            Warning(f"Not any positive point found in the prediction mask with threshold {thread}")

    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= (fp_masks | (all_correct & ~gt_masks)) * pred_masks_prob
    pts_noise[..., 1] *= fn_masks * gt_masks_prob
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def get_next_point_probabilistic(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors_probabilistic_mask(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")
