# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset_plus.box.box_segment_loader import (
    BoxSegmentLoader_trackinganygranularity
)

from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo, VOSRawDataset

from tqdm import tqdm
import PIL.Image as Image
from pycocotools.coco import COCO
from natsort import natsorted


class BoxRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,                 # image folder
        gt_folder,                  # ground truth folder (txt)
        png_folder=None,            # png folder
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        # is_palette=True,
        # single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.png_folder = png_folder
        self.sample_rate = sample_rate
        # self.is_palette = is_palette
        # self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        # self.video_names = sorted(
        #     [video_name for video_name in subset if video_name not in excluded_files]
        # )
        self.video_names = natsorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        # if self.single_object_mode:
        #     # single object mode
        #     self.video_names = sorted(
        #         [
        #             os.path.join(video_name, obj)
        #             for video_name in self.video_names
        #             for obj in os.listdir(os.path.join(self.gt_folder, video_name))
        #         ]
        #     )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.video_names)

class BoxRawDataset_trackinganygranularity(BoxRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        png_folder=None,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        # is_palette=True,
        # single_object_mode=True,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        super(BoxRawDataset_trackinganygranularity, self).__init__(
            img_folder=img_folder,
            gt_folder=gt_folder,
            png_folder=png_folder,
            file_list_txt=file_list_txt,
            excluded_videos_list_txt=excluded_videos_list_txt,
            sample_rate=sample_rate,
            truncate_video=truncate_video,
            frames_sampling_mult=frames_sampling_mult,
        )
    
    """
    Override the get_video method to return the BoxSegmentLoader_trackingnet object.
    """
    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        # if self.single_object_mode:
        #     video_frame_root = os.path.join(
        #         self.img_folder, os.path.dirname(video_name)
        #     )
        # else:
        #     video_frame_root = os.path.join(self.img_folder, video_name)
        video_frame_root = os.path.join(self.img_folder, video_name)

        # video_mask_root = os.path.join(self.gt_folder, video_name)
        video_mask_root = os.path.join(self.gt_folder, video_name) + '.txt'
        video_png_root = os.path.join(self.png_folder, video_name) if self.png_folder is not None else None

        # all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        all_frames = natsorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))

        # if self.is_palette:
        #     segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        # else:
        #     segment_loader = MultiplePNGSegmentLoader(
        #         video_mask_root, self.single_object_mode
        #     )
        segment_loader = BoxSegmentLoader_trackinganygranularity(video_mask_root, video_png_root)

        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader