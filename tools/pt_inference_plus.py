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
from PIL import Image
from sam2_plus.build_sam import build_sam2_video_predictor_plus

import logging
from benchmarks.pt_benchmark.datasets import construct_dataset

import cv2
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from training.utils_plus.visualization import color_palette

def _draw_point_single_image(
        img : np.ndarray, 
        pred_point: np.ndarray, 
        gt_point: np.ndarray, 
        pred_visibility: np.ndarray, 
        gt_visibility: np.ndarray
    ) -> np.ndarray: 
    """
    Draw the point on the image, pred point with 'circle', gt point with 'x', different color for different objects.
    @param 
        img: Narray, [H, W, 3]
    @param 
        pred_point: Narray, [N, 2], the output point, XY.
    @param 
        gt_point: Narray, [N, 2], the ground truth point, XY.
    @param
        pred_visibility: Narray, [N], the visibility of the pred point.
    @param
        gt_visibility: Narray, [N], the visibility of the gt point.
    Return:
        img: Narray, [H, W, 3]
    """
    N = pred_point.shape[0]
    for obj_id in range(N):
        color = color_palette[obj_id % len(color_palette)]
        # Draw pred point with 'circle'
        this_pred_point = pred_point[obj_id].astype(int)
        if pred_visibility[obj_id]:
            cv2.circle(img, (this_pred_point[0], this_pred_point[1]), 10, color, -1)
        # Draw gt point with 'x'            
        this_gt_point = gt_point[obj_id].astype(int)
        if gt_visibility[obj_id]:
            cv2.line(img, (this_gt_point[0] - 10, this_gt_point[1] - 10), (this_gt_point[0] + 10, this_gt_point[1] + 10), color, 5)
            cv2.line(img, (this_gt_point[0] + 10, this_gt_point[1] - 10), (this_gt_point[0] - 10, this_gt_point[1] + 10), color, 5)
    return img

def save_point_tracking_visualization(
    video_dir: str,
    gt_trajectory : np.ndarray,
    gt_visibility: np.ndarray,
    pred_trajectory: np.ndarray,
    pred_visibility: np.ndarray,
    output_dir: str,
    num_threads: int = 4,
):
    """
    draw ground truth and prediction point on the image, pred point with 'circle', gt point with 'x', different color for different objects.
    """
    # Check
    T = gt_trajectory.shape[0]
    assert gt_trajectory.shape == pred_trajectory.shape, f"gt_trajectory.shape={gt_trajectory.shape}, pred_trajectory.shape={pred_trajectory.shape}"
    frames_list = natsorted([p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", '.Png']])
    assert len(frames_list) == T, f"len(frames_list)={len(frames_list)}, gt_trajectory.shape[0]={gt_trajectory.shape[0]} \ncheck {video_dir}"
    
    os.makedirs(output_dir, exist_ok=True)

    def process_frame(frame_idx):
        frame_name = frames_list[frame_idx]
        img = Image.open(os.path.join(video_dir, frame_name))
        img = img.convert("RGB")
        img_np = np.array(img)
        img_with_point = _draw_point_single_image(img_np, pred_trajectory[frame_idx], gt_trajectory[frame_idx], pred_visibility[frame_idx], gt_visibility[frame_idx])
        img_with_point = Image.fromarray(img_with_point)
        img_with_point.save(os.path.join(output_dir, frame_name))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_frame, frame_idx) for frame_idx in range(T)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=T, desc="Save the visualization {}".format(output_dir)):
            pass
    
    # for idx in tqdm(range(T)):
    #     process_frame(idx)
    print(f"Save the visualization of the output pt to {output_dir}")

def load_visible_points_from_npz(
    input_points, input_visibles, frame_idx, allow_missing=False
):
    # Only load the visible points from the npz file
    if allow_missing and frame_idx >= len(input_points):
        return {}
    input_point, input_visible = input_points[frame_idx], input_visibles[frame_idx]
    object_ids = list(range(input_point.shape[0]))
    per_obj_point = {object_id: input_point[object_id] for object_id in object_ids if input_visible[object_id]}
    return per_obj_point

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def pt_inference(
    predictor,
    video_dir,
    frame_names,
    input_point,
    input_visible,
    output_pt_dir,
    video_name,
    radius=50,
    sigma=16,
    score_thresh=0.0,
    use_all_masks=False,
    lower_gpu_memory=False,
    skip_exist_result=False,
):
    num_frames, num_points = len(frame_names), input_point.shape[1]
    if skip_exist_result and os.path.exists(os.path.join(output_pt_dir, "result_point", f"{video_name}.npz")):
        pred_npz = np.load(os.path.join(output_pt_dir,"result_point", f"{video_name}.npz"))
        assert pred_npz['trajs_2d'].shape == (num_frames, num_points, 2) and pred_npz['visibs'].shape == (num_frames, num_points), f"Number of frames in the input and output directories are not the same for {video_name}. {pred_npz['trajs_2d'].shape}, {pred_npz['visibs'].shape}, {num_frames}, {num_points}"
        print(f"founded exist bbox from {os.path.join(output_pt_dir,'result_point', video_name.npz)}")
        return
    """Run Point Tracking inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    frame_names = natsorted(frame_names)
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=lower_gpu_memory, offload_video_to_cpu=lower_gpu_memory,
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # fetch mask inputs from input_mask_dir (either only mask for the first frame, or all available masks)
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [0]
    else:
        # use all mask files available in the input_mask_dir as the input masks
        input_frame_inds = [
            idx
            for idx, name in enumerate(frame_names)
            if torch.any(input_visible[idx])
        ]
        # check and make sure we got at least one input frame
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {video_dir=}. "
                "Please make sure the input masks are available in the correct format."
            )
        input_frame_inds = sorted(set(input_frame_inds))

    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        try:
            per_obj_input_point = load_visible_points_from_npz(
                input_points=input_point,
                input_visibles=input_visible,
                frame_idx=input_frame_idx,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_point)
            if len(object_ids_set) != num_points:
                raise RuntimeError(
                    f"In {video_name=}, Object '{set(range(num_points)) - set(object_ids_set)}' has no visible points in the first frame."
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                    f" Or check the input data {video_dir=} for more information."
                )
        for object_id, object_point in per_obj_input_point.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_points_and_generate_gaussian_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                points=object_point.unsqueeze(0).numpy(),
                labels=np.array([1]),
                radius=radius,
                sigma=sigma,
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
    point_array = -np.ones((num_frames, num_points, 2), dtype=np.float32)
    visible_array = np.zeros((num_frames, num_points), dtype=bool)
    for out_frame_idx, out_obj_ids, out_mask_logits, out_box_xyxys, out_obj_score_logits in predictor.propagate_in_video(
        inference_state
    ):
        for out_obj_id, out_mask_logit, out_obj_score_logit in zip(out_obj_ids, out_mask_logits, out_obj_score_logits):
            out_mask_logit, out_obj_score_logit = out_mask_logit.squeeze(0), out_obj_score_logit.squeeze(0)
            max_index = torch.argmax(out_mask_logit)
            max_score_y, max_score_x = torch.unravel_index(max_index, out_mask_logit.shape)
            point_array[out_frame_idx, out_obj_id] = np.array([max_score_x.cpu(), max_score_y.cpu()])
            visible_array[out_frame_idx, out_obj_id] = (out_obj_score_logit > score_thresh).cpu().numpy()

    # write the output masks as palette PNG files to output_mask_dir
    os.makedirs(os.path.join(output_pt_dir, "result_point"), exist_ok=True)
    np.savez(os.path.join(output_pt_dir, "result_point", f"{video_name}.npz"), trajs_2d=point_array, visibs=visible_array, size=(width, height))


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def pt_separate_inference_per_object(
    predictor,
    video_dir,
    frame_names,
    input_point,
    input_visible,
    output_pt_dir,
    video_name,
    radius=50,
    sigma=16,
    score_thresh=0.0,
    use_all_masks=False,
    lower_gpu_memory=False,
    skip_exist_result=False,
):
    num_frames, num_points = len(frame_names), input_point.shape[1]
    if skip_exist_result and os.path.exists(os.path.join(output_pt_dir, "result_point", f"{video_name}.npz")):
        pred_npz = np.load(os.path.join(output_pt_dir, "result_point", f"{video_name}.npz"))
        assert pred_npz['trajs_2d'].shape == (num_frames, num_points, 2) and pred_npz['visibs'].shape == (num_frames, num_points), f"Number of frames in the input and output directories are not the same for {video_name}. {pred_npz['trajs_2d'].shape}, {pred_npz['visibs'].shape}, {num_frames}, {num_points}"
        print(f"founded exist bbox from {os.path.join(output_pt_dir,'result_point', video_name.npz)}")
        return
    """
    Run Point Tracking inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    frame_names = natsorted(frame_names)
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=lower_gpu_memory, offload_video_to_cpu=lower_gpu_memory,
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        per_obj_input_point = load_visible_points_from_npz(
            input_points=input_point,
            input_visibles=input_visible,
            frame_idx=idx,
            allow_missing=True,
        )
        for object_id, object_point in per_obj_input_point.items():
            # if `use_all_masks=False`, we only use the first mask for each object
            if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                continue
            print(f"adding point from frame {idx} as input for {object_id=}")
            inputs_per_object[object_id][idx] = object_point

    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    point_array = -np.ones((num_frames, num_points, 2), dtype=np.float32)
    visible_array = np.zeros((num_frames, num_points), dtype=bool)
    if len(object_ids) != num_points:
        logging.warning(
            f"{video_name}: Object '{set(range(num_points)) - set(object_ids)}' has no visible points in the total video. "
            f"Skip their propagation and return these point=[-1, -1] and visible=False. Check {video_dir} for more information."
        )
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_points_and_generate_gaussian_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                points=inputs_per_object[object_id][input_frame_idx].unsqueeze(0).numpy(),
                labels=np.array([1]),
                radius=radius,
                sigma=sigma,
            )

        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits, out_box_xyxys, out_obj_score_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            out_mask_logits, out_obj_score_logits = out_mask_logits.squeeze(0).squeeze(0), out_obj_score_logits.squeeze(0).squeeze(0)
            max_index = torch.argmax(out_mask_logits)
            max_score_y, max_score_x = torch.unravel_index(max_index, out_mask_logits.shape)
            point_array[out_frame_idx, object_id] = np.array([max_score_x.cpu(), max_score_y.cpu()])
            visible_array[out_frame_idx, object_id] = (out_obj_score_logits > score_thresh).cpu().numpy()

    # write the output masks as palette PNG files to output_mask_dir
    os.makedirs(os.path.join(output_pt_dir, "result_point"), exist_ok=True)
    np.savez(os.path.join(output_pt_dir, "result_point", f"{video_name}.npz"), trajs_2d=point_array, visibs=visible_array, size=(width, height))


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
        "--output_pt_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=16,
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
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
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
        "--save_pt_visualization",
        action="store_true",
        help="Save the visualization of the output pt."
    )
    args = parser.parse_args()

    # NOTE: we set "non_overlap_masks" to false for point tracking to allow overlapping masks
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false")
    ]
    predictor = build_sam2_video_predictor_plus(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
        task='point'
    )

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )

    dataset = construct_dataset(args.dataset_name, args.dataset_dir)

    print(f"Running point tracking prediction on {len(dataset.sequence_list)} videos: \n {dataset.sequence_list}")

    for sequence in dataset.get_sequence():
        print(f"\n running on {sequence.video_name}")
        if not args.track_object_appearing_later_in_video:
            pt_inference(
                predictor=predictor,
                video_dir=sequence.video_dir,
                frame_names=sequence.frame_names,
                input_point=sequence.gt_trajectory,
                input_visible=sequence.gt_visibility,
                output_pt_dir=args.output_pt_dir,
                video_name=sequence.video_name,
                radius=args.radius,
                sigma=args.sigma,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                lower_gpu_memory=args.lower_gpu_memory,
                skip_exist_result=args.skip_exist_result,
            )
        else:
            pt_separate_inference_per_object(
                predictor=predictor,
                video_dir=sequence.video_dir,
                frame_names=sequence.frame_names,
                input_point=sequence.gt_trajectory,
                input_visible=sequence.gt_visibility,
                output_pt_dir=args.output_pt_dir,
                video_name=sequence.video_name,
                radius=args.radius,
                sigma=args.sigma,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                lower_gpu_memory=args.lower_gpu_memory,
                skip_exist_result=args.skip_exist_result,
            )
        if args.save_pt_visualization:
            pred_npz_file = os.path.join(args.output_pt_dir, "result_point", f"{sequence.video_name}.npz")
            pred_trajectory = np.load(pred_npz_file)["trajs_2d"]
            pred_visibility = np.load(pred_npz_file)["visibs"]
            save_point_tracking_visualization(
                video_dir=sequence.video_dir,
                gt_trajectory=sequence.gt_trajectory.cpu().numpy(),
                gt_visibility=sequence.gt_visibility.cpu().numpy(),
                pred_trajectory=pred_trajectory,
                pred_visibility=pred_visibility,
                output_dir=os.path.join(args.output_pt_dir, "visualization", sequence.video_name),
            )

    print(
        f"completed point tracking prediction on {len(dataset.sequence_list)} videos -- "
        f"output points saved to {args.output_pt_dir}"
    )

if __name__ == "__main__":
    main()