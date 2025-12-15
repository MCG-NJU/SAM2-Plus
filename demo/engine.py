import numpy as np
import torch
import cv2


# ============================================================
# ONE-TIME INTERACTION (PROMPT ONCE)
# ============================================================
def initialize_object(predictor, state, frame_idx, points, labels, obj_id=0):
    """
    Initialize object memory using user prompt.
    This MUST be called exactly once.
    """
    if len(points) == 0:
        raise RuntimeError("No points provided for initialization")

    res = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        points=np.array(points),
        labels=np.array(labels),
        obj_id=obj_id,
    )

    # SAM-2 output format:
    # res = (obj_id, obj_ids, mask_logits)
    logits = res[2]            # tensor (1, 1, H, W)

    logits = logits.squeeze().cpu()
    mask = (torch.sigmoid(logits) > 0.5).numpy().astype("uint8")

    return mask


# ============================================================
# PURE PROPAGATION (NO PROMPTS)
# ============================================================
def propagate_next(predictor, state):
    """
    Propagate object to the next frame using SAM-2 VOS generator.
    """
    # propagate_in_video returns a generator
    gen = predictor.propagate_in_video(state)

    # Get next frame result
    frame_idx, obj_ids, logits = next(gen)

    # logits shape: (num_objects, 1, H, W)
    logits = logits[0].squeeze().cpu()   # first object only

    mask = (torch.sigmoid(logits) > 0.5).numpy().astype("uint8")
    return mask


# ============================================================
# VISUALIZATION
# ============================================================
def overlay_mask(img, mask, alpha=0.5):
    """
    Overlay binary mask on image.
    """
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    overlay = img.copy()
    overlay[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
