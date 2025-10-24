import numpy as np
import torch


# Modified from https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/lib/utils/image.py#L118
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # FIX: square mask -> circle mask
    radius = min(m, n)
    mask = (x * x + y * y) <= radius * radius
    h[~mask] = 0

    return h

# Modified from https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/lib/utils/image.py#L126
def draw_umich_gaussian(heatmap, center, radius, k=1, sigma=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=sigma if sigma is not None else diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def generate_gaussian(center, img_size, radius=4, sigma=None):
    heatmap = np.zeros(img_size, dtype=np.float32)
    return draw_umich_gaussian(heatmap, center, radius, sigma=sigma)


def generate_single_point_heatmap(center, h, w):
    """
    center: x, y
    """
    heatmap = torch.zeros(h, w)
    x, y = int(center[0]), int(center[1])

    assert x >= 0 and x <= w and y >= 0 and y <= h, f"Invalid visible point: ({x}, {y}) for image size ({w}, {h})"
    if x == w or y ==h:
        print(f"Point ({x}, {y}) is out of bound. Clipping to ({min(x, w-1)}, {min(y, h-1)})")
        x = min(x, w-1)
        y = min(y, h-1)

    heatmap[y, x] = 1
    return heatmap


def calculate_center_point_from_heatmap(segment):
    """
    Calculate the center of the segment.
    @param segment: torch.Tensor, [H, W]
    @return: center: torch.Tensor[2], (x, y) format
    """
    non_zero_coords = torch.nonzero(segment > 0, as_tuple=False)
    center = torch.mean(non_zero_coords.float(), dim=0)
    # Swap y and x to return (x, y) format
    return torch.tensor([center[1], center[0]])


# https://discuss.pytorch.org/t/inverse-of-sigmoid-in-pytorch/14215
def revert_sigmoid(p, epsilon=1e-6):
    p = torch.clamp(p, epsilon, 1 - epsilon)
    return torch.log(p / (1 - p))