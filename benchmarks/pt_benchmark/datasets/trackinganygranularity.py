import os
import numpy as np
import torch
from natsort import natsorted
from benchmarks.pt_benchmark.datasets.data import PointTrackingSequence

class TrackingAnyGranularityDataset: 
    def __init__(self, data_dir, split='test'):
        super().__init__()
        assert split in ['test', 'valid'], 'Invalid split.'
        self.base_path = data_dir
        self.sequence_list = self._get_sequence_list(split)

    def get_sequence(self):
        for seq in self.sequence_list:
            yield self._construct_sequence(seq) 

    def _construct_sequence(self, sequence_name):
        npz_file_path = os.path.join(self.base_path, "Points", f"{sequence_name}.npz")
        trajs, visible, H, W = self._read_npz(npz_file_path)
        video_dir = os.path.join(self.base_path, "JPEGImages", sequence_name)

        return PointTrackingSequence(
            video_dir=video_dir,
            video_name=sequence_name,
            frame_names=natsorted([p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]),
            gt_trajectory=torch.tensor(trajs),
            gt_visibility=torch.tensor(visible),
            H=H,
            W=W
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open(f'{self.base_path}/../ImageSets/{split}.txt') as f:
            sequence_list = f.read().splitlines()
        return sequence_list

    def _read_npz(self, file_path):
        gt_data = np.load(file_path, allow_pickle=True)
        trajs = gt_data['trajs_2d'].astype(np.float32)    # ndarray [N_frames, N_points, 2], xyxy
        visible = gt_data['visibs'].astype(bool)    # ndarray [N_frames, N_points], bool
        W, H = gt_data['size']
        return trajs, visible, H, W