import os
import numpy as np
import torch
from benchmarks.pt_benchmark.datasets.data import PointTrackingSequence
from natsort import natsorted
from PIL import Image

class BADJADataset:
    def __init__(self, data_dir):
        self.task = 'point_tracking'
        self.data_dir = data_dir
        self.sequence_list = self._get_sequence_list()

    def get_sequence(self):
        for seq in self.sequence_list:
            yield self._construct_sequence(seq)

    def _construct_sequence(self, sequence_name):
        file_path = os.path.join(self.data_dir, "Points", f"{sequence_name}.npz")
        trajs, visible, H, W = self._read_npz(file_path)
        video_dir = os.path.join(self.data_dir, "PNGImages", sequence_name)
        annotations = os.path.join(self.data_dir, "Annotations", sequence_name)
        segmentation = self._load_segmentation(annotations) # T, 1, H, W

        return PointTrackingSequence(
            video_dir=video_dir,
            video_name=sequence_name,
            frame_names=natsorted([p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".png"]]),
            gt_trajectory=torch.tensor(trajs),
            gt_visibility=torch.tensor(visible),
            segmentation = segmentation, 
            H=H,
            W=W
        )

    def _get_sequence_list(self):
        return [
            "bear",
            "camel",
            "cows",
            "dog",
            "dog-agility",
            "horsejump-high",
            "horsejump-low",
            # "extra_videos_impala0",
            # "extra_videos_rs_dog",
            # "extra_videos_cat_jump", # Ignored
            # "DAVIS_tiger" # Ignored
        ]

    def _read_npz(self, file_path):
        gt_data = np.load(file_path, allow_pickle=True)
        trajs = gt_data['trajs_2d'].astype(np.float32)    # ndarray [N_frames, N_points, 2], xyxy
        visible = gt_data['visibs'].astype(bool)    # ndarray [N_frames, N_points], bool
        W, H = gt_data['size']
        return trajs, visible, H, W

    def _load_segmentation(self, mask_dir):
        mask_files = natsorted([p for p in os.listdir(mask_dir) if os.path.splitext(p)[-1] in [".png"]])
        masks = []
        for mask_file in mask_files:
            mask = np.array(Image.open(os.path.join(mask_dir, mask_file)))
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            masks.append(mask)
        return torch.stack(masks, dim=0)