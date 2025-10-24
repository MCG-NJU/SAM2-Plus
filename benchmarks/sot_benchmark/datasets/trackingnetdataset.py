import os
import numpy as np
from benchmarks.sot_benchmark.datasets.data import Sequence
from benchmarks.sot_benchmark.datasets.utils import load_text



class TrackingNetDataset:
    """ TrackingNet test set.
    """
    def __init__(self, data_path):
        super().__init__()
        self.base_path = data_path

        sets = 'TEST'
        if not isinstance(sets, (list, tuple)):
            if sets == 'TEST':
                sets = ['TEST']
            elif sets == 'TRAIN':
                sets = ['TRAIN_{}'.format(i) for i in range(5)]

        self.sequence_list = self._list_sequences(self.base_path, sets)

    def get_sequence(self):
        for set, seq_name in self.sequence_list:
            yield self._construct_sequence(set, seq_name)

    def _construct_sequence(self, set, sequence_name):
        anno_path = '{}/{}/anno/{}.txt'.format(self.base_path, set, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_dir = '{}/{}/frames/{}'.format(self.base_path, set, sequence_name)
        frames_name_list = [frame for frame in os.listdir(frames_dir) if frame.endswith(".jpg")]
        frames_name_list.sort(key=lambda f: int(f[:-4]))

        return Sequence(sequence_name, 
                        frames_dir,             
                        frames_name_list, 
                        'trackingnet',
                        ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _list_sequences(self, root, set_ids):
        sequence_list = []

        for s in set_ids:
            anno_dir = os.path.join(root, s, "anno")
            sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]

            sequence_list += sequences_cur_set

        return sequence_list
