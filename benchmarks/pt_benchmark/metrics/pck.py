import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# Copy from https://github.com/aharley/pips/blob/d8d77e2c675388e71c0729ba928f564de31a4c23/test_on_badja.py#L140C1-L159C19
def compute_pck_metrics(
    trajs_e,
    trajs_g,
    segs,
    visibles,
    pred_size_wh=None,
    gt_size_wh=None,
):
    metrics = {}
    B, S, N = visibles.shape
    assert(B==1)

    W_, H_ = pred_size_wh
    W, H = gt_size_wh
    sy = H_/H
    sx = W_/W

    # Reisze to pred_size
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # Resize segs to pred_size
    H_seg, W_seg = segs.shape[-2:]
    segs_ = segs.reshape(B*S, 1, H_seg, W_seg)
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    segs = segs_.reshape(B, S, 1, H_, W_)
    segs = (segs > 0).float()


    accs = []
    for s1 in range(1,S): # target frame
        for n in range(N):
            vis = visibles[0,s1,n]
            if vis > 0:
                coord_e = trajs_e[0,s1,n] # 2
                coord_g = trajs_g[0,s1,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,s1])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                # print_('correct', correct)
                accs.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    return metrics

def load_and_resize_masks(mask_dir, rgb_dir):
    mask_files = sorted(os.listdir(mask_dir))
    rgb_files = sorted(os.listdir(rgb_dir))
    masks = []

    for mask_file, rgb_file in zip(mask_files, rgb_files):
        mask_path = os.path.join(mask_dir, mask_file)
        rgb_path = os.path.join(rgb_dir, rgb_file)

        with Image.open(mask_path) as mask_img, Image.open(rgb_path) as rgb_img:
            mask_resized = mask_img.resize(rgb_img.size, Image.NEAREST)
            mask_array = np.array(mask_resized)
            masks.append(mask_array)

    masks = np.stack(masks, axis=0)  # Shape: [T, H, W]
    return masks

def evaluate_pck_from_npz(
    pred_npz_file,
    gt_npz_file,
    segs,
    force_resize_wh=None
):
    """
    Evaluate PCK (Percentage of Correct Keypoints) from npz files.

    Args:
        pred_npz_file (str): Path to the prediction .npz file.
        gt_npz_file (str): Path to the ground truth .npz file.
        segs (list): List of segmentations.
        force_resize_wh (tuple, optional): Tuple containing width and height to resize the images. Default is None.

    Returns:
        dict: Evaluation results.
    """
    pred_data = dict(np.load(pred_npz_file))
    gt_data = dict(np.load(gt_npz_file))

    if force_resize_wh is not None:
        # resize the groundtruth to the fixed size
        gt_size_wh = np.array(gt_data['size']) # w, h
        force_resize_wh = np.array(force_resize_wh) # w, h
        gt_scale_wh = force_resize_wh / gt_size_wh
        gt_data['trajs_2d'] = gt_data['trajs_2d'] * gt_scale_wh
        gt_data['size'] = force_resize_wh

        # resize the prediction to the fixed size
        pred_size_wh = np.array(pred_data['size'])
        pred_scale_wh = force_resize_wh / pred_size_wh
        pred_data['trajs_2d'] = pred_data['trajs_2d'] * pred_scale_wh
        pred_data['size'] = force_resize_wh

    trajs_e = torch.from_numpy(pred_data["trajs_2d"]).unsqueeze(0).float()
    trajs_g = torch.from_numpy(gt_data["trajs_2d"]).unsqueeze(0).float()
    visibles = torch.from_numpy(gt_data["visibs"]).unsqueeze(0).float()
    pred_size_wh = pred_data["size"]
    gt_size_wh = gt_data["size"]

    assert trajs_e.shape == trajs_g.shape, "Check the shape of the predicted and groundtruth trajectories"

    metrics = compute_pck_metrics(
        trajs_e,
        trajs_g,
        segs,
        visibles,
        pred_size_wh=pred_size_wh,
        gt_size_wh=gt_size_wh
    )
    return metrics

