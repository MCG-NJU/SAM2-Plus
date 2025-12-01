import numpy as np
import torch

import cv2

def run_segmentation(predictor,state,frame,points,labels,obj_id=0):
    res = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame,
        points=np.array(points),
        labels=np.array(labels),
        obj_id=obj_id
    )

    # SAM2 RETURN FORMAT → (id, mask_tensor, logits/info)
    mask = res[1]                       # <---- real mask
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()       # convert tensor to numpy

    mask = (mask > 0).astype("uint8")   # binarize properly
    return mask

def overlay_mask(img, mask):
    mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
    overlay = img.copy()
    overlay[mask==1] = (0,255,0)
    return cv2.addWeighted(overlay,0.5,img,0.5,0)
