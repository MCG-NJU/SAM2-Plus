import numpy as np
import torch
import torch.nn.functional as F
import cv2

def run_segmentation(predictor, state, frame, points, labels, obj_id=0):
    res = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame,
        points=np.array(points),
        labels=np.array(labels),
        obj_id=obj_id
    )

    # REAL MASK LOGITS → res[2]
    logits = res[2]                     # shape (1, 1, H, W)

    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().cpu()   # → (H, W)

    # Convert logits → probabilities
    probs = torch.sigmoid(logits)

    # Binary mask at 0.5 threshold
    mask = (probs > 0.5).numpy().astype("uint8")

    return mask

def overlay_mask(img, mask, alpha=0.5):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    overlay = img.copy()
    overlay[mask==1] = (0,255,0)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
