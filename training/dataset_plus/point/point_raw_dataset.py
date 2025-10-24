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

from training.dataset_plus.point.point_segment_loader import (
    PointSegmentLoader_trackinganygranularity,
)

from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo, VOSRawDataset

class PointRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        # is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        # self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [video_name for video_name in subset if os.path.isdir(os.path.join(self.img_folder, video_name))]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

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


class PointRawDataset_trackinganygranularity(PointRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        # is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        super(PointRawDataset_trackinganygranularity, self).__init__(
            img_folder=img_folder,
            gt_folder=gt_folder,
            file_list_txt=file_list_txt,
            excluded_videos_list_txt=excluded_videos_list_txt,
            sample_rate=sample_rate,
            # is_palette=is_palette,
            single_object_mode=single_object_mode,
            truncate_video=truncate_video,
            frames_sampling_mult=frames_sampling_mult,
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        # video_mask_root = os.path.join(self.gt_folder, video_name)
        # 
        # if self.is_palette:
        #     segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        # else:
        #     segment_loader = MultiplePNGSegmentLoader(
        #         video_mask_root, self.single_object_mode
        #     )
        segment_loader = PointSegmentLoader_trackinganygranularity(self.gt_folder, video_name)

        jpg_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        png_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.png")))
        assert (len(jpg_frames) > 0 or len(png_frames) > 0) and not (len(jpg_frames) > 0 and len(png_frames) > 0)
        
        all_frames = jpg_frames if len(jpg_frames) > 0 else png_frames
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader