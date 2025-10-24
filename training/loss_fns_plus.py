# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

import logging
from einops import rearrange
import torch.linalg as LA

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized
from training.loss_fns import sigmoid_focal_loss, dice_loss, iou_loss
from training.dataset_plus.box.utils import ciou_loss

def point_corrd_l1_loss(
    inputs,
    targets,
    num_objects,
    loss_on_multimask=False,
):
    """
    Compute the L1 loss
    Args:
        inputs: [N, M, 2], predicted coordinates
        targets: A float tensor with the same shape as inputs. normalized corrdinates
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    loss = F.l1_loss(inputs, targets, reduction='none').sum(dim=-1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


# Modify from https://github.com/facebookresearch/sam2/pull/376
def point_dice_loss_with_soft_label(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Reference:
        Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels.
                Wang, Z. et. al. MICCAI 2023.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()

    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        # numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        # numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    difference = LA.vector_norm(inputs - targets, ord=1, dim=-1)
    numerator = denominator - difference
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def point_iou_loss_with_unscale_and_threshold(
    inputs, targets, pred_ious, num_objects, threshold=0.0, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    inputs = inputs.sigmoid()
    
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > threshold   # pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > threshold    # gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def box_iou_loss_given_actual_ious(
    actual_ious, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        actual_ious: A float tensor containing the actual IoUs scores per mask
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    # assert inputs.dim() == 4 and targets.dim() == 4
    # pred_mask = inputs.flatten(2) > 0
    # gt_mask = targets.flatten(2) > 0
    # area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    # area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    # actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous_point(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,

        size=1024,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert 'loss_l1' in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

        self.size = size

        with torch.no_grad():
            indice = torch.arange(0, size).view(-1, 1)
            # generate mesh-grid
            self.coord_x = indice.repeat((size, 1)).view((size * size,)).float().cuda()
            self.coord_y = indice.repeat((1, size)).view((size * size,)).float().cuda()

    def forward(self, outs_batch: List[Dict], target_masks_batch: torch.Tensor, target_points_batch: torch.Tensor, target_visibles_batch: torch.Tensor):
        assert len(outs_batch) == len(target_masks_batch)
        assert len(outs_batch) == len(target_points_batch)
        assert len(outs_batch) == len(target_visibles_batch)
        num_objects = torch.tensor(
            (target_masks_batch.shape[1]), device=target_masks_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, target_masks, target_points, target_visibles in zip(outs_batch, target_masks_batch, target_points_batch, target_visibles_batch):
            cur_losses = self._forward(outs, target_masks, target_points, target_visibles, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, target_masks: torch.Tensor, target_points: torch.Tensor, target_visibles: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = target_masks.unsqueeze(1).float()
        target_points_norm = target_points.unsqueeze(1).float() / self.size
        target_visibles = target_visibles.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        assert target_points_norm.dim() == 3  # [N, 1, 2]
        assert target_visibles.dim() == 2  # [N, 1]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_l1": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, target_points_norm, target_visibles, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, target_points_norm, target_visibles, ious, num_objects, object_score_logits
    ):
        src_masks_flat = rearrange(src_masks, 'b m h w -> (b m) h w')
        src_points_norm_flat = self.soft_argmax(src_masks_flat, coord_norm=True)
        src_points_norm = rearrange(src_points_norm_flat, '(b m) n -> b m n', b=src_masks.shape[0], m=src_masks.shape[1], n=2)

        target_masks = target_masks.expand_as(src_masks)
        target_points_norm = target_points_norm.expand_as(src_points_norm)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = point_dice_loss_with_soft_label(   # dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        # get L1 loss on all output points in a prediction step
        loss_multicorrdL1 = point_corrd_l1_loss(
            src_points_norm,
            target_points_norm,
            num_objects,
            loss_on_multimask=True,
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            # target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
            #     ..., None
            # ].float()
            target_obj = target_visibles.to(dtype=src_masks.dtype, device=src_masks.device)
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = point_iou_loss_with_unscale_and_threshold(  # iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            threshold=0.2,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multicorrdL1.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
                + loss_multicorrdL1 * self.weight_dict["loss_l1"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            loss_coordL1 = loss_multicorrdL1[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_coordL1 = loss_multicorrdL1
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_coordL1 = loss_coordL1 * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_l1"] += loss_coordL1.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
    
    # def get_argmax(self, target, coord_norm):
    #     bs_obj = target.shape[0]
    #     max_indices = torch.argmax(target.view(bs_obj, -1), dim=1)
    #     x, y = max_indices % self.size, max_indices // self.size
    #     return torch.stack((x, y), dim=1).float() / (self.size if coord_norm else 1.0)

    def soft_argmax(self, score_map, coord_norm=True):
        score_vec = score_map.view((-1, self.size * self.size))  # (batch, size * size)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return torch.stack((exp_x, exp_y), dim=1) / (self.size if coord_norm else 1.0)


class MultiStepMultiMasksAndIous_box(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,

        size=1024,
        pred_mask=False,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_ciou" in self.weight_dict
        assert 'loss_l1' in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

        self.size = size
        self.pred_mask = pred_mask

    def forward(self, outs_batch: List[Dict], target_masks_batch: torch.Tensor, target_boxes_batch: torch.Tensor, target_visibles_batch: torch.Tensor):
        assert len(outs_batch) == len(target_masks_batch)
        assert len(outs_batch) == len(target_boxes_batch)
        assert len(outs_batch) == len(target_visibles_batch)
        num_objects = torch.tensor(
            (target_boxes_batch.shape[1]), device=target_boxes_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, target_masks, target_boxes, target_visibles in zip(outs_batch, target_masks_batch, target_boxes_batch, target_visibles_batch):
            cur_losses = self._forward(outs, target_masks, target_boxes, target_visibles, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, target_masks: torch.Tensor, target_boxes: torch.Tensor, target_visibles: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = target_masks.unsqueeze(1).float()
        target_boxes_norm = target_boxes.unsqueeze(1).float() / self.size
        target_visibles = target_visibles.unsqueeze(1)
        assert target_masks.dim() == 4  # [N, 1, H, W]
        assert target_boxes_norm.dim() == 3  # [N, 1, 4]
        assert target_visibles.dim() == 2  # [N, 1]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        src_boxes_norm_list = outputs["multistep_pred_multiboxes_xyxy_norm"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(src_boxes_norm_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_ciou": 0, "loss_l1": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, src_boxes_norm, ious, object_score_logits in zip(
            src_masks_list, src_boxes_norm_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, src_boxes_norm, target_masks, target_boxes_norm, target_visibles, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, src_boxes_norm, target_masks, target_boxes_norm, target_visibles, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        target_boxes_norm = target_boxes_norm.expand_as(src_boxes_norm)

        # if src_boxes_norm[:, :, 0] >= src_boxes_norm[:, :, 2] or src_boxes_norm[:, :, 1] >= src_boxes_norm[:, :, 3]:
        #     logging.warning("There are some unvalid boxes in the prediction.")
        if self.pred_mask:
            # get focal, dice and iou loss on all output masks in a prediction step
            loss_multimask = sigmoid_focal_loss(
                src_masks,
                target_masks,
                num_objects,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                loss_on_multimask=True,
            )
            loss_multidice = dice_loss(
                src_masks, target_masks, num_objects, loss_on_multimask=True
            )
        else:
            loss_multimask = torch.zeros(
                src_boxes_norm.shape[0], src_boxes_norm.shape[1], dtype=src_boxes_norm.dtype, device=src_boxes_norm.device
            )
            loss_multidice = torch.zeros(
                src_boxes_norm.shape[0], src_boxes_norm.shape[1], dtype=src_boxes_norm.dtype, device=src_boxes_norm.device
            )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            # target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
            #     ..., None
            # ].float()
            target_obj = target_visibles.to(dtype=src_boxes_norm.dtype, device=src_boxes_norm.device)
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )
        
        bs_obj, n, _ = src_boxes_norm.shape

        # there are some not visible objects in the target, we set its gt to [0, 0, 1, 1] for unified loss computation, but we ignore the loss by ` * target_obj`
        not_visible = ~target_obj.bool().repeat(1, target_boxes_norm.shape[1])
        target_boxes_norm = target_boxes_norm.clone()
        target_boxes_norm[not_visible] = torch.tensor([0, 0, 1, 1], dtype=target_boxes_norm.dtype, device=target_boxes_norm.device)

        loss_multi_ciou, actual_ious = ciou_loss(
            src_boxes_norm.reshape(-1, 4),
            target_boxes_norm.reshape(-1, 4),
            mean_batch=False
        )
        loss_multi_ciou, actual_ious = loss_multi_ciou.reshape(bs_obj, n), actual_ious.reshape(bs_obj, n)

        loss_multi_l1 = F.l1_loss(
            src_boxes_norm, target_boxes_norm, reduction='none'
        ).sum(dim=-1)

        loss_multiiou = box_iou_loss_given_actual_ious(    # iou_loss(
            actual_ious,
            # src_masks,
            # target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )# [bs_obj, num_output]
        if self.pred_mask:
            _loss_multiiou = iou_loss(
                src_masks,
                target_masks,
                ious,
                num_objects,
                loss_on_multimask=True,
                use_l1_loss=self.iou_use_l1_loss,
            )
            loss_multiiou = loss_multiiou + _loss_multiiou

        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multi_ciou.dim() == 2
        assert loss_multi_l1.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multi_ciou.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
                + loss_multi_ciou * self.weight_dict["loss_ciou"]
                + loss_multi_l1 * self.weight_dict["loss_l1"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            loss_ciou = loss_multi_ciou[batch_inds, best_loss_inds].unsqueeze(1)
            loss_l1 = loss_multi_l1[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_ciou = loss_multi_ciou
            loss_l1 = loss_multi_l1
            loss_iou = loss_multiiou

        # backprop focal and dice loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        # backprop ciou, l1 and iou loss only if obj present
        loss_ciou = loss_ciou * target_obj
        loss_l1 = loss_l1 * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_ciou"] += loss_ciou.sum()
        losses["loss_l1"] += loss_multi_l1.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
