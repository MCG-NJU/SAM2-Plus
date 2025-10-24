from benchmarks.pt_benchmark.metrics.Jaccard import evaluate_jaccard_from_npz
from benchmarks.pt_benchmark.metrics.pck import evaluate_pck_from_npz, load_and_resize_masks
import torch
import numpy as np
import os
import json
import argparse
from benchmarks.pt_benchmark.datasets import construct_dataset

class PointTrackingScore_base:
    def __init__(self):
        pass

    def cal_metrics(self,):
        raise NotImplementedError()

    def _average_metrics(self, dicts):
        """
        Compute the average of metrics across multiple dictionaries.
        """
        avg_dict = {}
        keys = dicts[0].keys()
        for key in keys:
            avg_dict[key] = np.mean([d[key] for d in dicts], axis=0)
            if isinstance(avg_dict[key], np.ndarray):
                avg_dict[key] = avg_dict[key].item()
        return avg_dict

    def score_for_dataset(self, video_names, pred_dir, gt_dir):
        raise NotImplementedError()

class PointTrackingScore_aj(PointTrackingScore_base):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def cal_metrics(self, pred_npz_file, gt_npz_file, scale=256.0):
        """
        Calculate metrics for the given prediction and ground truth npz files.

        Args:
            pred_npz_file (str): Path to the prediction .npz file. Contains 'trajs_2d' (ndarray [N_frames, N_points, 2], xyxy), 'visibs' (ndarray [N_frames, N_points], bool), and 'size' (WH).
            gt_npz_file (str): Path to the ground truth .npz file. Contains 'trajs_2d' (ndarray [N_frames, N_points, 2], xyxy), 'visibs' (ndarray [N_frames, N_points], bool), and 'size' (WH).
            scale (float, optional): A float value representing the scaling factor. Default is 256.0.

        Returns:
            dict: Calculated metrics.
        """
        metrics = evaluate_jaccard_from_npz(pred_npz_file, gt_npz_file, scale)
        return metrics 
    
    def score_for_dataset(self, video_names, pred_dir, gt_dir):
        metrics_res_all = []
        for video_name in video_names:
            gt_npz_file = os.path.join(gt_dir, f"{video_name}.npz")
            pred_npz_file = os.path.join(pred_dir, f"{video_name}.npz")
            metrics = self.cal_metrics(pred_npz_file, gt_npz_file)
            metrics_res_all.append(metrics)
        metrics_res_avg = self._average_metrics(metrics_res_all)
        return metrics_res_avg

class PointTrackingScore_pck(PointTrackingScore_base):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def cal_metrics(self, pred_npz_file, gt_npz_file, gt_mask_resized):
        """
        Calculate metrics for the given prediction and ground truth npz files.

        Args:
            pred_npz_file (str): Path to the prediction .npz file. Contains 'trajs_2d' (ndarray [N_frames, N_points, 2], xyxy), 'visibs' (ndarray [N_frames, N_points], bool), and 'size' (WH).
            gt_npz_file (str): Path to the ground truth .npz file. Contains 'trajs_2d' (ndarray [N_frames, N_points, 2], xyxy), 'visibs' (ndarray [N_frames, N_points], bool), and 'size' (WH).
            gt_mask_resized (ndarray): Ground truth mask resized to the same size as the raw RGB image.

        Returns:
            dict: Calculated metrics.
        """
        metrics = evaluate_pck_from_npz(pred_npz_file, 
                                        gt_npz_file, 
                                        gt_mask_resized,
                                        force_resize_wh=(512, 320),
                                        )
        return metrics

    def score_for_dataset(self, video_names, pred_dir, gt_dir, detailed=True):
        """
        Calculate scores for the entire dataset.

        Args:
            video_names (list): List of video names to be evaluated.
            pred_dir (str): Directory containing the prediction files.
            gt_dir (str): Directory containing the ground truth files.
            detailed (bool, optional): Whether to return detailed scores. Default is True.

        Returns:
            dict: Scores for the dataset.
        """
        metrics_res_all = []
        mask_dir = gt_dir.replace('Points', 'Annotations')
        raw_img_dir = gt_dir.replace('Points', 'PNGImages')

        metrics_res_detailed = {}
        for video_name in video_names:
            gt_mask_resized = load_and_resize_masks(
                os.path.join(mask_dir, video_name),
                os.path.join(raw_img_dir, video_name),
            )
            gt_mask_resized = torch.from_numpy(gt_mask_resized).unsqueeze(0)
            gt_npz_file = os.path.join(gt_dir, f"{video_name}.npz")
            pred_npz_file = os.path.join(pred_dir, f"{video_name}.npz")
            metrics = self.cal_metrics(pred_npz_file, gt_npz_file, gt_mask_resized)
            metrics_res_all.append(metrics)
            metrics_res_detailed[f'pck_{video_name}'] = metrics.get('pck')
        metrics_res_avg = self._average_metrics(metrics_res_all)
        if detailed:
            metrics_res_avg.update(metrics_res_detailed)
        return metrics_res_avg

def run_score(dataset_name, metrics_type, dataset_dir, pred_npz_dir, groundtruth_npz_dir):
    assert metrics_type in ['tapvid', 'pck']
    dataset = construct_dataset(dataset_name, dataset_dir)
    if metrics_type == 'tapvid':
        metric_calculator = PointTrackingScore_aj(dataset_name)
        metrics = metric_calculator.score_for_dataset(dataset.sequence_list, pred_npz_dir, groundtruth_npz_dir)
    elif metrics_type == 'pck':
        metric_calculator = PointTrackingScore_pck(dataset_name)
        metrics = metric_calculator.score_for_dataset(dataset.sequence_list, pred_npz_dir, groundtruth_npz_dir)
    else:
        raise ValueError(f"Unknown metrics type: {metrics_type}")
    
    metrics_save_dir = os.path.dirname(os.path.normpath(pred_npz_dir))
    print(f"Metrics: {metrics}")
    with open(f"{metrics_save_dir}/{metrics_type}_result.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_save_dir}/{metrics_type}_result.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--metrics_type",
        type=str,
        default='tapvid',
        help="type of metrics to calculate. Default: 'tapvid'",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="directory containing the dataset",
    )
    parser.add_argument(
        "--groundtruth_npz_dir",
        type=str,
        required=False,
        help="directory containing ground truth npz files",
    )
    parser.add_argument(
        "--pred_npz_dir",
        type=str,
        required=True,
        help="directory containing predicted npz files",
    )
    args = parser.parse_args()

    metrics_info = {
        "BADJA": "pck",
        "tracking_any_granularity_val": "tapvid",
        "tracking_any_granularity_test": "tapvid",
    }
    standard_metrics = metrics_info.get(args.dataset_name)
    if standard_metrics != args.metrics_type:
        print(f"Warning: The standard metrics for {args.dataset_name} is {standard_metrics}, but you are using {args.metrics_type}")

    run_score(args.dataset_name, args.metrics_type, args.dataset_dir, args.pred_npz_dir, args.groundtruth_npz_dir)
