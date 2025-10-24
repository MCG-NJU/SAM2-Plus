import numpy as np
from collections import OrderedDict
from benchmarks.sot_benchmark.datasets.utils import imread_indexed


class Sequence:
    """
    Class for the sequence in an evaluation.
    Warp from MixFormer/lib/test/evaluation/data.py
    """
    def __init__(self, 
                 video_name, 
                 frames_path, 
                 frames, 
                 dataset, 
                 ground_truth_rect, 
                 ground_truth_seg=None, 
                 init_data=None,
                 object_class=None, 
                 target_visible=None, 
                 object_ids=None, 
                 multiobj_mode=False):
        """
        @param video_name: name of the sequence.  eg.`airplane/airplane-1`. We need it to save the results to out_fold/name
        @param frames_path: path to the frames. eg. `datasets/Single_Object_Tracking/lasot/airplane/airplane-1/img`
        @param frames: list of frame names. eg. ['0001.jpg', '0002.jpg', ...]
        @param dataset: name of the dataset. eg. `lasot`
        @param ground_truth_rect: ground truth bounding box. numpy array of shape (num_frames, 4) or dict of numpy array of shape (num_frames, 4) 
        @object_class: class of the object. eg. `airplane`
        @target_visible: list of boolean values indicating whether the target is visible in each frame. eg. [True, True, False, ...]
        """
        
        self.video_name = video_name
        self.frames_path = frames_path
        self.frames = frames
        self.dataset = dataset
        self.ground_truth_rect = ground_truth_rect
        if len(self.ground_truth_rect) == 0 :
            raise ValueError(f"ground_truth_rect is empty, check {frames_path}")
        self.ground_truth_seg = ground_truth_seg
        self.object_class = object_class
        self.target_visible = target_visible
        self.object_ids = object_ids
        self.multiobj_mode = multiobj_mode
        assert self.multiobj_mode == False, "Multi-object mode is not supported yet."
        self.init_data = self._construct_init_data(init_data)
        self._ensure_start_frame()

    def _ensure_start_frame(self):
        # Ensure start frame is 0
        start_frame = min(list(self.init_data.keys()))
        if start_frame > 0:
            self.frames = self.frames[start_frame:]
            if self.ground_truth_rect is not None:
                if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                    for obj_id, gt in self.ground_truth_rect.items():
                        self.ground_truth_rect[obj_id] = gt[start_frame:,:]
                else:
                    self.ground_truth_rect = self.ground_truth_rect[start_frame:,:]
            if self.ground_truth_seg is not None:
                self.ground_truth_seg = self.ground_truth_seg[start_frame:]
                assert len(self.frames) == len(self.ground_truth_seg)

            if self.target_visible is not None:
                self.target_visible = self.target_visible[start_frame:]
            self.init_data = {frame-start_frame: val for frame, val in self.init_data.items()}

    def _construct_init_data(self, init_data):
        if init_data is not None:
            if not self.multiobj_mode:
                assert self.object_ids is None or len(self.object_ids) == 1
                for frame, init_val in init_data.items():
                    if 'bbox' in init_val and isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = init_val['bbox'][self.object_ids[0]]
            # convert to list
            for frame, init_val in init_data.items():
                if 'bbox' in init_val:
                    if isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = OrderedDict({obj_id: list(init) for obj_id, init in init_val['bbox'].items()})
                    else:
                        init_val['bbox'] = list(init_val['bbox'])
        else:
            init_data = {0: dict()}     # Assume start from frame 0

            if self.object_ids is not None:
                init_data[0]['object_ids'] = self.object_ids

            if self.ground_truth_rect is not None:
                if self.multiobj_mode:
                    assert isinstance(self.ground_truth_rect, (dict, OrderedDict))
                    init_data[0]['bbox'] = OrderedDict({obj_id: list(gt[0,:]) for obj_id, gt in self.ground_truth_rect.items()})
                else:
                    assert self.object_ids is None or len(self.object_ids) == 1
                    if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                        init_data[0]['bbox'] = list(self.ground_truth_rect[self.object_ids[0]][0, :])
                    else:
                        try:
                            init_data[0]['bbox'] = list(self.ground_truth_rect[0,:])
                        except:
                            import pdb; pdb.set_trace()

            if self.ground_truth_seg is not None:
                init_data[0]['mask'] = self.ground_truth_seg[0]

        return init_data

    def object_init_data(self, frame_num=None) -> dict:
        if frame_num is None:
            frame_num = 0
        if frame_num not in self.init_data:
            return dict()

        init_data = dict()
        for key, val in self.init_data[frame_num].items():
            if val is None:
                continue
            init_data['init_'+key] = val

        if 'init_mask' in init_data and init_data['init_mask'] is not None:
            anno = imread_indexed(init_data['init_mask'])
            if not self.multiobj_mode and self.object_ids is not None:
                assert len(self.object_ids) == 1
                anno = (anno == int(self.object_ids[0])).astype(np.uint8)
            init_data['init_mask'] = anno

        if self.object_ids is not None:
            init_data['object_ids'] = self.object_ids
            init_data['sequence_object_ids'] = self.object_ids

        return init_data

    def get_frames(self):
        return self.frames
    
    def get_frames_path(self):
        return self.frames_path
    
    def get_init_bbox(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_bbox')
    
    def get_name(self):
        return self.video_name
    
    def __repr__(self):
        return f"{self.__class__.__name__} | {self.dataset} | {self.video_name} | {len(self.frames)} frames"
