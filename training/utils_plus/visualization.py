# from training.utils.data_utils import BatchedVideoDatapoint, VideoDatapoint
# from training.utils_plus.data_utils import BatchedVideoDatapointWithBoxesPoints, BoxObject, PointObject, Object

import os
import torch
import matplotlib.pyplot as plt
# from PIL import Image
from typing import Optional, Union, List
import numpy as np
import cv2
from torchvision import utils as vutils
from matplotlib import pyplot as plt
# from PIL import ImageDraw
# from dataclasses import dataclass, field


color_palette = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),# Silver
    (128, 128, 128),# Gray
    (0, 0, 0)       # Black
]

def _denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean, device=tensor.device).reshape(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).reshape(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)
    
def draw_dashed_rectangle(img, top_left, bottom_right, color, thickness=2, dash_length=10):
    """
    Draw a dashed rectangle on the image.
    @param img: Input image
    @param top_left: Top-left corner coordinates (x, y)
    @param bottom_right: Bottom-right corner coordinates (x, y)
    @param color: Color (B, G, R)
    @param thickness: Line thickness
    @param dash_length: Length of each dash segment
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw top border
    for i in range(x1, x2, dash_length * 2):
        pt1 = (i, y1)
        pt2 = (min(i + dash_length, x2), y1)
        cv2.line(img, pt1, pt2, color, thickness)

    # Draw bottom border
    for i in range(x1, x2, dash_length * 2):
        pt1 = (i, y2)
        pt2 = (min(i + dash_length, x2), y2)
        cv2.line(img, pt1, pt2, color, thickness)

    # Draw left border
    for i in range(y1, y2, dash_length * 2):
        pt1 = (x1, i)
        pt2 = (x1, min(i + dash_length, y2))
        cv2.line(img, pt1, pt2, color, thickness)

    # Draw right border
    for i in range(y1, y2, dash_length * 2):
        pt1 = (x2, i)
        pt2 = (x2, min(i + dash_length, y2))
        cv2.line(img, pt1, pt2, color, thickness)


class TensorboardVisualizer:
    def __init__(self):
        self.colors = color_palette    

    def log_visualize_multi_task_io(
            self,
            task: str,
            input_imgs: torch.Tensor,
            gt_segments: torch.Tensor, gt_boxes: torch.Tensor = None, gt_points: torch.Tensor = None,
            output_segments: torch.Tensor = None, output_boxes: torch.Tensor = None, output_points: torch.Tensor = None,
            prompt_points: list = None, prompt_masks: list = None,
        ):
        """
        @param task: str, the task name, [mask, box, point]
        @param step: int, the step number
        @param input_imgs: Tensor, [T, 3, H, W]

        @param gt_segments: Tensor, [T, N, H, W], the ground truth segments. N is the number of objects.
        @param gt_boxes: Tensor, [T, N, 4], the ground truth boxes. XYXY format. N is 1.
        @param gt_points: Tensor, [T, N, 2], the ground truth points. XY format. N is the number of objects.
        
        @param output_segments: Tensor, [T, N, H, W], the output segments. N is the number of objects.
        @param output_boxes: Tensor, [T, N, 4], the output boxes. XYXY format. 
        @param output_points: Tensor, [T, N, 2], the output points. XY format.


        @param prompt_points: List of Tensors, Len = T, each is {'point_coords': [1, P, 2], 'point_labels': [1, P]} or None, where P is the number of the prompt points.
        @param prompt_masks: List of Tensor/None, Len = T, each is [N, 1, H, W] or None

        """
        # 1. denormalize the input images
        if task == 'point' and output_points is None:
            output_points = self._prepare_pred_point(output_segments) # [T, N, 2]
        input_imgs = _denormalize(input_imgs) # [T, 3, H, W]
        prompt_imgs = self._prepare_prompt_img(input_imgs, prompt_masks, prompt_points) # [T, 3, H, W]

        # 2. Reshape tensors to [T*N, C, H, W] for input images and [T*N, H, W] for segments
        T, N, H, W = gt_segments.shape

        if task == "mask":
            score_thresh = 0.5
            diff_pred_gt = 1 - torch.eq(output_segments > score_thresh, gt_segments).int() # [T, N, H, W]
            diff_pred_gt_mask_img = self._draw_segment(input_imgs, diff_pred_gt, obj_num = N) # [T, 3, H, W]
            output_mask_img_grid = vutils.make_grid(diff_pred_gt_mask_img, nrow=T, normalize=True, scale_each=True).cpu()
        
        gt_segments = self._mask_to_rgb(gt_segments, score_thresh= 0.5)
        output_segments = self._mask_to_rgb(output_segments, score_thresh= 0.5)
        
        # 3. Create grid images
        input_grid = vutils.make_grid(input_imgs, nrow=T, normalize=True, scale_each=True).cpu()
        prompt_imgs_grid = vutils.make_grid(prompt_imgs, nrow=T, normalize=True, scale_each=True).cpu()
        gt_segments_grid = vutils.make_grid(gt_segments, nrow=T, normalize=True, scale_each=True).cpu()
        output_segments_grid = vutils.make_grid(output_segments, nrow=T, normalize=True, scale_each=True).cpu()

        if task == "box":
            output_boxes_img = self._draw_box(input_imgs, output_boxes, gt_boxes)
            output_boxes_img_grid = vutils.make_grid(output_boxes_img, nrow=T*N, normalize=True, scale_each=True).cpu()
        
        if task == "point":
            output_points_img = self._draw_point(input_imgs, output_points, gt_points)
            output_points_img_grid = vutils.make_grid(output_points_img, nrow=T, normalize=True, scale_each=True).cpu()
        

        # 4. Combine the grid images
        if task == "mask":
            combined_grid = torch.cat([output_mask_img_grid, prompt_imgs_grid, gt_segments_grid, output_segments_grid], dim=1)
        if task == "box":
            combined_grid = torch.cat([output_boxes_img_grid, prompt_imgs_grid,  gt_segments_grid, output_segments_grid], dim=1)
        if task == "point":
            combined_grid = torch.cat([output_points_img_grid, prompt_imgs_grid,  gt_segments_grid, output_segments_grid], dim=1)

        return combined_grid

    def _overlay_mask(self, img, mask, alpha=0.5):
        return img * (1 - alpha) + mask * alpha

    def _prepare_prompt_img(self, input_img, prompt_masks: List[Optional[torch.Tensor]], prompt_points: List[Optional[torch.Tensor]]):
        """
        @param input_img: Tensor, [T, 3, H, W],
        @param prompt_masks: List of Tensor/None, Len = T, each is [N, 1, H, W] or None
        @param prompt_points: List of Tensors, Len = T, each is [N, M, 2] or None, M is the number of the prompt points.

        Return:
            prompted_img: Tensor, [T, 3, H, W]
        """
        res = []
        T = input_img.shape[0]

        for t in range(T):
            raw_img = input_img[t] # [3, H, W]
            raw_img_np = (raw_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            raw_img_np = cv2.cvtColor(raw_img_np, cv2.COLOR_RGB2BGR)
            
            if prompt_masks[t] is not None:
                alpha = 0.5
                mask_colored = np.zeros_like(raw_img_np)
                N = prompt_masks[t].shape[0]
                for obj_id in range(N):
                    mask_np = prompt_masks[t][obj_id].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                    mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
                    color = np.array(self.colors[obj_id % len(self.colors)], dtype=np.uint8)
                    mask_colored = np.where(mask_rgb > 0, color, mask_colored)
                raw_img_np = cv2.addWeighted(raw_img_np, 1 - alpha, mask_colored, alpha, 0)

            if prompt_points[t] is not None:
                N = prompt_points[t]['point_coords'].shape[0]
                P = prompt_points[t]['point_coords'].shape[1]
                for obj_id in range(N):
                    color = self.colors[obj_id % len(self.colors)]
                    labels = prompt_points[t]['point_labels'][obj_id]
                    if (labels == torch.tensor([2,3])).all(): # Box
                        x1, y1 = prompt_points[t]['point_coords'][obj_id, 0].numpy().astype(np.int32)
                        x2, y2 = prompt_points[t]['point_coords'][obj_id, 1].numpy().astype(np.int32)
                        cv2.rectangle(raw_img_np, (x1, y1), (x2, y2), color, thickness=10)
                    elif torch.isin(labels, torch.tensor([0, 1])).all(): # Poi/Neg Points
                        for point_id in range(P):
                            x, y = prompt_points[t]['point_coords'][obj_id, point_id]
                            x, y = int(x), int(y)
                            if labels[point_id] == 0:
                                cv2.circle(raw_img_np, (x, y), 10, color, 5) # Negative point, hollow circle
                            else:
                                cv2.circle(raw_img_np, (x, y), 10, color, -1) # Positive point, filled circle
                    else:
                        raise ValueError(f"Invalid label {labels}, it should be [0, 1]* or [2,3], but got {labels}")
            
            raw_img_np = cv2.cvtColor(raw_img_np, cv2.COLOR_BGR2RGB)
            res.append(torch.from_numpy(raw_img_np).permute(2, 0, 1).float() / 255)

        return torch.stack(res, dim=0)
    
    def _convert_point_prompt_to_box_prompt(self, prompt_points):
        """"
        @param prompt_points: List of Tensors, Len = T, each is [N, M, 2] or None, M is the number of the prompt points.
        """
    
    def _prepare_pred_point(self, gt_segments):
        """
        @param gt_segments: Tensor, [T, N, H, W], the ground truth segments.
        Return:
            pred_point: [T, N, 2], the output points, int.
        """
        T, N, H, W = gt_segments.shape
        max_point_idx = torch.argmax(gt_segments.reshape(T*N, -1), dim=-1)
        y_coords, x_coords = max_point_idx // W, max_point_idx % W
        pred_point = torch.stack([x_coords, y_coords], dim=-1)
        pred_point = pred_point.reshape(T, N, 2)
        return pred_point

    
    def _draw_box(self, input_img, pred_box, gt_boxes):
        """
        @param input_img: Tensor, [T, 3, H, W]
        @param pred_box: [T, N, 4], the output boxes, XYXY.
        @param gt_boxes: [T, N, 4], the ground truth boxes, XYXY.
        Return:
            img_with_box: Tensor, [T, 3, H, W]
        """
        res = []
        T, N = pred_box.shape[:2]
        for t in range(T):
            img = self._draw_box_single(input_img[t], pred_box[t], gt_boxes[t])
            res.append(img.unsqueeze(0))
        return torch.cat(res, dim=0)

    def _draw_box_single(self, img, pred_box, gt_box):
        """
        @param img: Tensor, [3, H, W]
        @param pred_box: [N, 4], the output box, XYXY.
        @param gt_box: [N, 4], the ground truth box, XYXY.
        """
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred_box = pred_box.cpu().numpy().astype(np.int32)
        gt_box = gt_box.cpu().numpy().astype(np.int32)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        N = pred_box.shape[0]
        for obj_idx in range(N):
            color = self.colors[obj_idx % len(self.colors)]
            if pred_box is not None:
                cv2.rectangle(img_np, (pred_box[obj_idx, 0], pred_box[obj_idx, 1]), (pred_box[obj_idx, 2], pred_box[obj_idx, 3]), color=color, thickness=5)
            if gt_box is not None:
                draw_dashed_rectangle(img_np, (gt_box[obj_idx, 0], gt_box[obj_idx, 1]), (gt_box[obj_idx, 2], gt_box[obj_idx, 3]), color=color, thickness=5)
                # cv2.rectangle(img_np, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=color, thickness=5)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255
        return img_tensor
    

    def _draw_point(self, input_img, pred_point, gt_point):
        """
        @param input_img: Tensor, [T, 3, H, W]
        @param pred_point: [T, N, 2], the output points, int.
        @param gt_point: [T, N, 2], the ground truth points, int, if value is [-1,-1], represent no point.

        Return:
            img_with_point: Tensor, [T, 3, H, W]
        """
        res = []
        for i in range(input_img.shape[0]):
            img = self._draw_point_single(input_img[i], pred_point[i], gt_point[i])
            res.append(img.unsqueeze(0))
        return torch.cat(res, dim=0)
    
    def _draw_point_single(self, img, pred_point, gt_point):
        """
        Draw the point on the image, pred point with 'circle', gt point with 'x', different color for different objects.
        @param 
            img: Tensor, [3, H, W]
        @param 
            pred_point: [N, 2], the output point, XY.
        @param 
            gt_point: [N, 2], the ground truth point, XY.
        Return:
            img_tensor: Tensor, [3, H, W]
        """
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        N = pred_point.shape[0]
        for obj_id in range(N):
            color = self.colors[obj_id % len(self.colors)]
            # Draw pred point with 'circle'
            this_pred_point = pred_point[obj_id].cpu().numpy().astype(np.int32)
            cv2.circle(img_np, (this_pred_point[0], this_pred_point[1]), 10, color, -1)
            # Draw gt point with 'x'            
            this_gt_point = gt_point[obj_id].cpu().numpy().astype(np.int32)
            cv2.line(img_np, (this_gt_point[0] - 10, this_gt_point[1] - 10), (this_gt_point[0] + 10, this_gt_point[1] + 10), color, 5)
            cv2.line(img_np, (this_gt_point[0] + 10, this_gt_point[1] - 10), (this_gt_point[0] - 10, this_gt_point[1] + 10), color, 5)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255
        return img_tensor

    def _draw_segment(self, input_img, segment_prediction_mask, obj_num, score_thresh=0.5):
        """
        @param input_img: Tensor, [T, 3, H, W]
        @param segment_prediction_mask: [T, N, H, W], the output segments, float.

        Return:
            img_with_mask: Tensor, [T, 3, H, W]
        """
        T, N, H, W = segment_prediction_mask.shape
        segment_prediction_mask = segment_prediction_mask > score_thresh  # [T, N, H, W]
        res = []
        for t in range(T):
            img_np = (input_img[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            mask_colored = np.zeros_like(img_np)
            for obj_id in range(N):
                mask_np = segment_prediction_mask[t, obj_id].cpu().numpy().astype(np.uint8) * 255
                mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
                color = np.array(self.colors[obj_id % len(self.colors)], dtype=np.uint8)
                mask_colored = np.where(mask_rgb > 0, color, mask_colored)
            alpha = 0.5
            img_np = cv2.addWeighted(img_np, 1 - alpha, mask_colored, alpha, 0)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255
            res.append(img_tensor.unsqueeze(0))
        return torch.cat(res, dim=0)

    def _mask_to_rgb(self, mask, score_thresh=0.5):
        """[T,N,H,W] -> [T,3,H,W]"""

        T, N, H, W = mask.shape
        mask = mask > score_thresh
        res = []
        for t in range(T):
            all_objs_mask_colored = np.zeros((H,W,3), dtype=np.uint8)
            for obj_id in range(N):
                this_mask = (mask[t, obj_id].cpu().numpy() * 255).astype(np.uint8)
                this_mask = cv2.cvtColor(this_mask, cv2.COLOR_GRAY2BGR)
                color = np.array(self.colors[obj_id % len(self.colors)], dtype=np.uint8)
                all_objs_mask_colored = np.where(this_mask > 0, color, all_objs_mask_colored)
            all_objs_mask_colored = cv2.cvtColor(all_objs_mask_colored, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(all_objs_mask_colored).permute(2, 0, 1).float() / 255
            res.append(img_tensor.unsqueeze(0))
        return torch.cat(res, dim=0)


    # def _convert_matplotlib_to_tensor(self, fig):
    #     fig.canvas.draw()
    #     w, h = fig.canvas.get_width_height()
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    #     image = np.transpose(image, (2, 0, 1))
    #     return torch.from_numpy(image)