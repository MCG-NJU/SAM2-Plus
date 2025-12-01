import torch
from sam2.build_sam import build_sam2_video_predictor

def load_sam2(model_cfg, checkpoint, device="cuda"):
    """
    Load SAM2 Video Predictor model.
    """
    device = device if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return predictor, device


def initialize_inference(predictor, video_dir):
    """
    Create a fresh inference state for a given video/frames folder.
    """
    state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(state)
    return state
