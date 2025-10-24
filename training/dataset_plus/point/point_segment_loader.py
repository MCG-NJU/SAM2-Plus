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


class PointSegmentLoader:
    def __init__(self, video_gt_root, video_name):
        """
        Args:
            video_gt_root (str): Path of the gt files, each file contains the ground truth data of a whole video.
            video_name (int | str): The ID of the frame to load.
        """
        self.trajs, self.visible = self.get_sequence_info(video_gt_root, video_name)

    def get_sequence_info(self, video_gt_root, video_name):
        '''
        trajs: ndarray [N_frames, N_points, 2], xyxy
        visible: ndarray [N_frames, N_points], bool
        H, W: int, height and width of the image
        '''
        raise NotImplementedError

    def load(self, frame_id):
        """
        Load the ground truth data for a specific frame.

        Args:
            frame_id: int, define the mask path
        Return:
            points: dict
            visibles: dict
        """

        object_id = range(self.trajs.shape[1])
        
        points, visibles = {}, {}
        for i in object_id:
            gt_points = self.trajs[frame_id, i] # array[2]: x,y
            visible = self.visible[frame_id, i] # np.bool

            points[i] = torch.from_numpy(gt_points) # Tensor[2,]
            visibles[i] = bool(visible)             # bool

        return points, visibles

    def __len__(self):
        return


class PointSegmentLoader_trackinganygranularity(PointSegmentLoader):
    def __init__(self, video_gt_root, video_name):
        super().__init__(video_gt_root, video_name)

    def get_sequence_info(self, video_gt_root, video_name):
        file_path = os.path.join(video_gt_root, f"{video_name}.npz")
        gt_data = np.load(file_path, allow_pickle=True)
        trajs = gt_data['trajs_2d'].astype(np.float32)    # ndarray [N_frames, N_points, 2], xyxy
        visible = gt_data['visibs'].astype(bool)    # ndarray [N_frames, N_points], bool
        # W, H = gt_data['size']
        return trajs, visible