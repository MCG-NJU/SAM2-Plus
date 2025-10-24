import os
import numpy as np
from benchmarks.sot_benchmark.datasets.data import Sequence
from benchmarks.sot_benchmark.datasets.utils import load_text


class TrackingAnyGranularityDataset:    
    """
    TrackingAnything dataset class.
    """
    def __init__(self, data_path, split='test'):
        super().__init__()
        assert split in ['test', 'valid'], 'Invalid split.'
        self.base_path = data_path
        self.sequence_list = self._get_sequence_list(split)

    def get_sequence(self):
        """
        Returns a generator that yields constructed sequences from the sequence list.
        Yields:
            dict: A constructed sequence from the sequence list.
        """
        for seq in self.sequence_list:
            yield self._construct_sequence(seq) 

    def _construct_sequence(self, sequence_name):

        anno_path = os.path.join(self.base_path, "Boxes", f"{sequence_name}.txt")
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        visible_path = os.path.join(self.base_path, "Visibles", f"{sequence_name}.txt")
        target_visible = load_text(str(visible_path), delimiter=',', dtype=np.int32)

        frames_path = f"{self.base_path}/JPEGImages/{sequence_name}"
        frames_name_list = [f for f in os.listdir(frames_path)]
        frames_name_list = sorted(frames_name_list)

        # assert len(frames_name_list) == len(ground_truth_rect), 'Number of frames and annotations do not match.'
        return Sequence(sequence_name, 
                    frames_path,
                    frames_name_list, 
                    'tracking_anything', 
                    ground_truth_rect.reshape(-1, 4), 
                    target_visible=target_visible
                    )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open(f'{self.base_path}/../ImageSets/{split}.txt') as f:
            sequence_list = f.read().splitlines()
        return sequence_list
