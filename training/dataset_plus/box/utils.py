import math
import torch
from torchvision.ops.boxes import box_area
import numpy as np
from torch.nn import functional as F
from torch import nn

def np_box_clamp_xywh(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, None)


def np_box_xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    assert x.shape[-1] == 4
    x1, y1, w, h = np.split(x, x.shape[-1], axis=-1)
    b = [x1, y1, x1 + w, y1 + h]
    return np.concatenate(b, axis=-1)


def np_box_xyxy_to_xywh(x: np.ndarray) -> np.ndarray:
    assert x.shape[-1] == 4
    x1, y1, x2, y2 = np.split(x, x.shape[-1], axis=-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return np.concatenate(b, axis=-1)


def np_box_xywh_to_cxcy(x: np.ndarray) -> np.ndarray:
    assert x.shape[-1] == 4
    x1, y1, w, h = np.split(x, x.shape[-1], axis=-1)
    center_point = [x1 + w // 2, y1 + h // 2]
    return np.concatenate(center_point, axis=-1)


# Modified from https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html#masks_to_boxes
def np_masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] np.array containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 <= x2`` and ``0 <= y1 <= y2``.

    .. warning::

        In most cases the output will guarantee ``x1 < x2`` and ``y1 < y2``. But
        if the input is degenerate, e.g. if a mask is a single row or a single
        column, then the output may have x1 = x2 or y1 = y2.

    Args:
        masks (np.array [N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        np.array[N, 4]: bounding boxes
    """
    assert masks.ndim == 3

    n = masks.shape[0]

    bounding_boxes = np.zeros((n, 4), dtype=np.int32)

    for index, mask in enumerate(masks):
        if not np.any(mask):
            continue

        y, x = np.where(mask != 0)

        bounding_boxes[index, 0] = np.min(x)
        bounding_boxes[index, 1] = np.min(y)
        bounding_boxes[index, 2] = np.max(x)
        bounding_boxes[index, 3] = np.max(y)

    return bounding_boxes


def box_to_mask(box, height, width, target_visible:torch.Tensor, normalized=False):
    """
    @param box: Tensor [B, 4] or [4] in [x, y, x, y] format. 
    @param height: height of the mask
    @param width: width of the mask
    @param target_visible: Tensor [B] or boolean, whether the target is visible
    @param normalized: whether the box is normalized
    @return:
        Tensor [B, height, width]
    """

    box = box.clone()   # avoid modifying the original box
    
    if box.dim() == 1:
        box = box.unsqueeze(0)
        target_visible = target_visible.unsqueeze(0)
    B,_ = box.shape
    this_gt = torch.zeros(size=(B, height, width), device=box.device)
    for i in range(B):
        if not target_visible[i]:
            continue

        if normalized:
            box[i][0] = box[i][0] * width
            box[i][1] = box[i][1] * height
            box[i][2] = box[i][2] * width
            box[i][3] = box[i][3] * height

        # clip within image
        box[i][0] = torch.clip(box[i][0], 0, width)  # x_min
        box[i][1] = torch.clip(box[i][1], 0, height) # y_min
        box[i][2] = torch.clip(box[i][2], 0, width)  # x_max
        box[i][3] = torch.clip(box[i][3], 0, height) # y_max

        this_gt[i][int(box[i][1]):int(box[i][3]), int(box[i][0]):int(box[i][2])] = 1

    return this_gt


# =========== Copied from https://github.com/MCG-NJU/MixFormer/blob/main/lib/utils/box_ops.py =========== #

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou



def ciou_loss(bboxes1, bboxes2, mean_batch=True):   # modified 1e-7 for numerical stability
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2.
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2.
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2.
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2.

    inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

    c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / (c_diag + 1e-7)
    iou = inter_area / (union+1e-7)
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / (h2+1e-7)) - torch.atan(w1 / (h1+1e-7))), 2)
    with torch.no_grad():
        S = (iou>0.5).float()
        alpha= S*v/(1-iou+v+1e-7)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return torch.mean(1-cious) if mean_batch else (1-cious), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]
