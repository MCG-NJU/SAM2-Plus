# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from natsort import natsorted
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2_plus.build_sam import build_sam2_video_predictor_plus

from tools.sot_inference import save_masks_and_boxes_to_dir, save_boxes_to_dir

import gc
import logging
from benchmarks.sot_benchmark.datasets import construct_dataset
from training.dataset_plus.box.utils import np_box_xywh_to_xyxy, np_box_xyxy_to_xywh, np_masks_to_boxes, np_box_clamp_xywh
from tools.vos_inference import DAVIS_PALETTE, put_per_obj_mask




@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def sot_inference(
    predictor,
    video_dir,
    frame_names,
    input_box_xywh,
    output_box_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    lower_gpu_memory=False,
    save_box_visualization=False,
    skip_exist_result=False,
):
    """Run SOT inference on a single video with the given predictor.
    @param predictor: the predictor object
    @param video_dir: the directory containing video frames
    @param frame_names: the list of frame names (the absolute path of each frame is video_dir/frame_name.jpg)
    @param input_box_xywh: the initial box in the format of [xmin, ymin, width, height]
    @param output_box_dir: the directory to save the output bbox
    @param video_name: the name of the video
    @param score_thresh: the threshold for the output mask logits
    @param skip_exist_result: whether to load the existing result from output_box_dir

    The result will be saved to output_box_dir
    ├── result_bbox
    │   ├── GOT-10k_Val_000001.txt
    │   └── GOT-10k_Val_000002.txt
    └── visualization
        ├── GOT-10k_Val_000001
            ├── 00000001.png
            ├── 00000002.png
    """
    if skip_exist_result:
        result_bbox_path = os.path.join(output_box_dir, "result_bbox", video_name + ".txt")
        if os.path.exists(result_bbox_path):
            result_bbox = np.loadtxt(result_bbox_path, dtype=np.int32)
            assert result_bbox.shape[0] == len(frame_names), f"result_bbox.shape[0] != len(frame_names)"
            print(f"founded exist bbox from {result_bbox_path}")
            return result_bbox

    # load the video frames and initialize the inference state on this video
    frame_names = natsorted(frame_names)
    video_dir_frame_names = natsorted([
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
    ])
    if frame_names == video_dir_frame_names:
        start_frame_idx = None
        max_frame_num_to_track = None
    else:
        print(f"In {video_dir} Mismatched \n {len(frame_names)} != {len(video_dir_frame_names)}")
        start_frame_idx = video_dir_frame_names.index(frame_names[0])
        max_frame_num_to_track = len(frame_names) - 1
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=lower_gpu_memory, offload_video_to_cpu=lower_gpu_memory,
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # fetch box inputs
    assert not use_all_masks, "Only support single object now"
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [start_frame_idx if start_frame_idx is not None else 0]

    # add those input boxes to SAM 2 inference state before propagation
    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        try:
            per_obj_input_box_xyxy = {1: np_box_xywh_to_xyxy(np.array(input_box_xywh))}
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_box_xyxy)
        for object_id, object_box_xyxy in per_obj_input_box_xyxy.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                box=object_box_xyxy,
            )

    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no object ids on {input_frame_inds=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )
    # run propagation throughout the video and collect the results in a dict
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    video_boxes_xywh = {}  # video_boxes_xyxy contains the per-frame bounding box results
    for out_frame_idx, out_obj_ids, out_mask_logits, output_box_xyxy, out_obj_score_logits in predictor.propagate_in_video(
        inference_state=inference_state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track,
    ):
        if torch.any(output_box_xyxy[:,:,0] >= output_box_xyxy[:,:,2]) or torch.any(output_box_xyxy[:,:,1] >= output_box_xyxy[:,:,3]):
            logging.warning(f"Invalid box prediction: {output_box_xyxy}")
    
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask
        per_obj_output_box_xywh = {
            out_obj_id: np_box_clamp_xywh(np_box_xyxy_to_xywh(output_box_xyxy[i].cpu().numpy()))
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_boxes_xywh[out_frame_idx] = per_obj_output_box_xywh
    
    # write the output masks as palette PNG files to output_mask_dir
    if save_box_visualization:
        for out_frame_idx, per_obj_output_box_xywh in video_boxes_xywh.items():
            print(f"save visualization for {video_name}/{frame_names[out_frame_idx]} to {output_box_dir}")
            save_masks_and_boxes_to_dir(
                output_mask_dir=os.path.join(output_box_dir, "visualization"),
                video_name=video_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=video_segments[out_frame_idx],
                per_obj_output_box_xywh=per_obj_output_box_xywh,
                height=height,
                width=width,
                per_obj_png_file=per_obj_png_file,
                output_palette=output_palette,
            )
    
    # save bbox as txt file
    save_boxes_to_dir(
        output_bbox_dir=os.path.join(output_box_dir, "result_bbox"),
        video_name=video_name,
        video_boxes_xywh=video_boxes_xywh,
    )
    
    inference_state = predictor.reset_state(inference_state)
    del inference_state
    del video_segments
    del video_boxes_xywh
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--output_box_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
    )
    parser.add_argument(
        "--lower_gpu_memory",
        action="store_true",
        help="whether to offload video frames to CPU memory to save GPU memory",
    )
    parser.add_argument(
        "--skip_exist_result",
        action="store_true",
        help="Skip the video if the number of frames in the input and output directories are the same."
    )
    parser.add_argument(
        "--save_box_visualization",
        action="store_true",
        help="Save the visualization of the output box."
    )
    args = parser.parse_args()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor_plus(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
        task='box'
    )

    dataset = construct_dataset(args.dataset_name, args.dataset_dir)

    print(f"running SOT prediction on {len(dataset.sequence_list)} videos:\n{dataset.sequence_list}")

    for dataset_sequence in dataset.get_sequence():
        sot_inference(
            predictor=predictor,
            video_dir=dataset_sequence.get_frames_path(),
            frame_names=dataset_sequence.get_frames(),
            input_box_xywh=dataset_sequence.get_init_bbox(),
            output_box_dir=args.output_box_dir,
            video_name=dataset_sequence.get_name(),
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
            lower_gpu_memory=args.lower_gpu_memory,
            save_box_visualization=args.save_box_visualization,
            skip_exist_result=args.skip_exist_result,
        )

    print(
        f"completed SOT prediction on {len(dataset.sequence_list)} videos -- "
        f"output boxes saved to {args.output_box_dir}"
    )


if __name__ == "__main__":
    main()