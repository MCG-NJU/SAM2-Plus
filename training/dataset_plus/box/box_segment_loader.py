# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image as PILImage

try:
    from pycocotools import mask as mask_utils
except:
    pass

import pandas
import csv 
import random
from training.dataset_plus.box.utils import box_xywh_to_xyxy

class BoxSegmentLoader:
    def __init__(self, video_box_gt_path, video_png_root):
        """
        Args:
            video_box_gt_path (str): Path of the box ground truth file
            video_png_root: the folder contains all the masks stored in png
        """
        sequence_info = self.get_sequence_info(video_box_gt_path)
        video_gt_bbox_xywh, video_target_visible = sequence_info['bbox'], sequence_info['visible']
        self.video_gt_bbox_xyxy = box_xywh_to_xyxy(video_gt_bbox_xywh)
        self.video_gt_bbox_xyxy[:, :2] = torch.clamp(self.video_gt_bbox_xyxy[:, :2], min=0.0)   # clamp the x1y1 >= 0
        self.video_target_visible = video_target_visible & (self.video_gt_bbox_xyxy[:, 2] > 0) & (self.video_gt_bbox_xyxy[:, 3] > 0)    # x2y2 > 0
        self.get_sequence_png(video_png_root)

    def get_sequence_info(self, video_box_gt_path):
        """
        Given the path of the box ground truth file, return the bbox and visible info of the whole video.
        Args:
            video_box_gt_path (str): Path of the box ground truth file
        Return:
            {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
        """
        raise NotImplementedError()

    def _read_bb_anno(self, seq_path):
        raise NotImplementedError()
    
    def _read_target_visible(self, seq_path):
        raise NotImplementedError()

    def get_sequence_png(self, video_png_root):
        raise NotImplementedError()

    def load(self, frame_id):
        """
        Load the ground truth data for a specific frame.

        Args:
            frame_id (int): The ID of the frame to load.

        Returns:
            box_xyxys: dict
            box_visibles: dict
            box_pngs: dict
        """
        # check the path
        if self.frame_id_to_png_filename is not None:
            mask_path = os.path.join(
                self.video_png_root, self.frame_id_to_png_filename[frame_id]
            )

            # load the mask
            masks = PILImage.open(mask_path).convert("P")
            masks = np.array(masks)

            object_id = pd.unique(masks.flatten())
            object_id = object_id[object_id != 0]  # remove background (0)
            assert (len(object_id) == 1 and object_id[0] == 1) or len(object_id) == 0
        else:
            masks = None

        object_id = [1]
        
        box_xyxys, box_visibles, box_pngs = {}, {}, {}
        for i in object_id:
            bbox_xyxy = self.video_gt_bbox_xyxy[frame_id]   # Tensor(4,)
            visible = self.video_target_visible[frame_id]   # Tensor(1,)

            box_xyxys[i] = bbox_xyxy            # Tensor[4,]
            box_visibles[i] = bool(visible)     # bool
            
            if masks is not None:
                bs = masks == i
                box_pngs[i] = torch.from_numpy(bs)
            else:
                box_pngs[i] = None

        return box_xyxys, box_visibles, box_pngs

    def __len__(self):
        return


class BoxSegmentLoader_trackinganygranularity(BoxSegmentLoader):
    def __init__(self, video_box_gt_path, video_png_root):
        super().__init__(video_box_gt_path, video_png_root)
    
    # def get_sequence_info(self, seq_id):
    #     bbox = self._read_bb_anno(seq_id)

    #     valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    #     visible = valid.clone().byte()
    #     return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_sequence_info(self, video_box_gt_path):
        bbox = self._read_bb_anno(video_box_gt_path)
        
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(video_box_gt_path.replace('Boxes', 'Visibles')) & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    # def _read_bb_anno(self, seq_id):
    #     set_id = self.sequence_list[seq_id][0]
    #     vid_name = self.sequence_list[seq_id][1]
    #     bb_anno_file = os.path.join(self.root, "TRAIN_" + str(set_id), "anno", vid_name + ".txt")
    #     gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
    #                          low_memory=False).values
    #     return torch.tensor(gt)

    def _read_bb_anno(self, seq_path):
        # Read ground-truth bbox
        gt = pandas.read_csv(seq_path, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        with open(seq_path, 'r', newline='') as f:
            target_visible = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        
        return target_visible
    
    def get_sequence_png(self, video_png_root):
        self.video_png_root = video_png_root
        if video_png_root is None:
            self.frame_id_to_png_filename = None
            return
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        png_filenames = os.listdir(self.video_png_root)
        self.frame_id_to_png_filename = {}
        for filename in png_filenames:
            frame_id, _ = os.path.splitext(filename)
            self.frame_id_to_png_filename[int(frame_id)] = filename