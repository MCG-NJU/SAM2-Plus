import os
import numpy as np
from benchmarks.sot_benchmark.datasets.data import Sequence
from benchmarks.sot_benchmark.datasets.utils import load_text


class GOT10KDataset:
    """ GOT-10k dataset.
    """
    def __init__(self, data_path, split = 'val'):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(data_path, split)
        else:
            self.base_path = os.path.join(data_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence(self):
        for s in self.sequence_list:
            yield self._construct_sequence(s)

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = os.path.join(self.base_path, sequence_name)

        frames_name_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frames_name_list.sort(key=lambda f: int(f[:-4]))

        return Sequence(sequence_name, 
                        frames_path,
                        frames_name_list, 
                        'got10k', 
                        ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        if split == 'ltrval':
            raise NotImplementedError("ltrval split not implemented")

        return sequence_list
