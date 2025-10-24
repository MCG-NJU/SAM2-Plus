import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

@dataclass(eq=False)
class PointTrackingSequence:
    """
    Dataclass for SAM2 video inference data.
    """
    video_dir: str
    video_name: str
    frame_names: List[str] # T
    gt_trajectory: torch.Tensor  # T, N, 2
    gt_visibility: torch.Tensor  # T, N
    # Optional data
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    segmentation: Optional[torch.Tensor] = None  # T, 1, H, W
    H : Optional[int] = None
    W : Optional[int] = None
