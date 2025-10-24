# Copy from https://github.com/google-deepmind/perception_test/blob/0e19d3c434df815ae47afc060890de34e16f3ce4/baselines/single_point_tracking.ipynb#L25
from typing import Any, Dict, Mapping
import numpy as np


# from https://github.com/facebookresearch/co-tracker/blob/main/cotracker/evaluation/core/eval_utils.py#L12
def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        if count_visible_points != 0:
           frac_correct = count_correct / count_visible_points
        else:
           print("Warning: pts_within_thresh computation division by zero")
           frac_correct = np.array([0])
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        if (gt_positives + false_positives) != 0:
          jaccard = true_positives / (gt_positives + false_positives)
        else:
          print("Warning: jaccard computation division by zero")
          jaccard = np.array([0])
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


# from https://github.com/google-deepmind/perception_test/blob/0e19d3c434df815ae47afc060890de34e16f3ce4/baselines/single_point_tracking.ipynb#L25
def evaluate(results: Dict[str, Any], label_dict: Dict[str, Any],
                scale: float = 256.0) -> float:
  """Calculates the Average Jaccard for each video in the results.

  Args:
    results: A dictionary containing the results for each video.
      The keys are video IDs, and the values are dictionaries with
      'point_tracking' information.
    label_dict: A dictionary containing the ground truth labels for each video.
      The keys are video IDs, and the values are dictionaries with
      'point_tracking' information.
    scale: A float value representing the scaling factor (default is 256.0).

  Returns:
    The average Jaccard across all videos.

  Raises:
    AssertionError: If the lengths of predicted tracks and ground truth tracks
      do not match.

  """
  avg_jacs = []
  # static_avg_jacs = []
  # moving_avg_jacs = []
  for video_id, video_item in results.items():
    pred_tracks = video_item['point_tracking']
    gt_tracks = label_dict[video_id]['point_tracking']
    num_frames = label_dict[video_id]['metadata']['num_frames']
    assert len(pred_tracks) == len(gt_tracks)
    num_tracks = len(pred_tracks)

    query_points = np.zeros((1, num_tracks, 3))
    gt_occluded = np.ones((1, num_tracks, num_frames))
    gt_points = np.zeros((1, num_tracks, num_frames, 2))
    pred_occluded = np.ones((1, num_tracks, num_frames))
    pred_points = np.zeros((1, num_tracks, num_frames, 2))

    for track_idx, pred_track in enumerate(pred_tracks):
      gt_track = gt_tracks[pred_track['id']]
      gt_track_points = np.array(gt_track['points']).T
      pred_track_points = np.array(pred_track['points']).T

      start_point = gt_track_points[0]
      start_frame_id = gt_track['frame_ids'][0]
      query_points[0, track_idx, 0] = start_frame_id
      query_points[0, track_idx, 1:] = start_point
      gt_occluded[0, track_idx][gt_track['frame_ids']] = 0
      pred_occluded[0, track_idx] = 0

      gt_points[0, track_idx][gt_track['frame_ids']] = gt_track_points
      pred_points[0, track_idx, :, :][pred_track['frame_ids']] = (
          pred_track_points
      )

    gt_points *= scale
    pred_points *= scale

    metrics = compute_tapvid_metrics(query_points, gt_occluded,
                                     gt_points, pred_occluded,
                                     pred_points, 'first')
    avg_jacs.append(metrics['average_jaccard'])

    # if label_dict[video_id]['metadata']['is_camera_moving']:
    #   moving_avg_jacs.append(metrics['average_jaccard'])
    # else:
    #   static_avg_jacs.append(metrics['average_jaccard'])

  average_jaccard = np.mean(avg_jacs)
  # static_average_jaccard = np.mean(static_avg_jacs)
  # moving_average_jaccard = np.mean(moving_avg_jacs)
  print(f'Average Jaccard across all videos: {average_jaccard}')
  # print(f'Average Jaccard across static videos: {static_average_jaccard}')
  # print(f'Average Jaccard across moving videos: {moving_average_jaccard}')
  return average_jaccard


def evaluate_jaccard_from_npz(npz_file_path: str, gt_npz_file_path: str, scale: float = 256.0) -> float:
    """
    Calculates the Average Jaccard for each video from an npz file.

    Args:
        npz_file_path (str): Path to the npz file containing the results. {'trajs_2d': ndarray, 'visibs': ndarray, 'size': ndarray, w,h format}
        gt_npz_file_path (str): Path to the npz file containing the ground truth labels. {'trajs_2d': ndarray, 'visibs': ndarray, 'size': ndarray, w,h format}
        scale (float, optional): A float value representing the scaling factor. Default is 256.0.

    Returns:
        float: The average Jaccard for one video.
    """
    # Load data from npz files
    pred_data = np.load(npz_file_path)
    pred_trajs = pred_data['trajs_2d'].astype(int) # ndarray [N_frames, N_points, 2], xy
    pred_visible = pred_data['visibs'].astype(bool) # ndarray [N_frames, N_points], bool
    pred_size = pred_data['size'].reshape(1,1,2) #(1，1，2),  w, h format
    pred_trajs_norm = pred_trajs / pred_size

    gt_data = np.load(gt_npz_file_path)
    gt_trajs = gt_data['trajs_2d'].astype(int) # ndarray [N_frames, N_points, 2], xyxy
    gt_visible = gt_data['visibs'].astype(bool) # ndarray [N_frames, N_points], bool

    gt_size = gt_data['size'].reshape(1,1,2) #(1，1，2),  w, h format
    gt_trajs_norm = gt_trajs / gt_size

    assert pred_trajs_norm.shape == gt_trajs_norm.shape
    assert pred_visible.shape == gt_visible.shape
    num_frames, num_tracks, _ = pred_trajs_norm.shape

    query_points = np.zeros((1, num_tracks, 3))
    gt_occluded = np.ones((1, num_tracks, num_frames))
    gt_points = np.zeros((1, num_tracks, num_frames, 2))
    pred_occluded = np.ones((1, num_tracks, num_frames))
    pred_points = np.zeros((1, num_tracks, num_frames, 2))

    for track_idx in range(num_tracks):
        gt_track_points = gt_trajs_norm[:, track_idx, :]

        start_point = gt_track_points[0]
        # Find the first visible frame
        start_frame_id = np.where(gt_visible[:, track_idx])[0][0] if np.any(gt_visible[:, track_idx]) else (num_frames - 1)
        query_points[0, track_idx, 0] = start_frame_id
        query_points[0, track_idx, 1:] = start_point.T  # Y, X （Though it is not used in metrics calculation）
        gt_occluded[0, track_idx] = ~gt_visible[:, track_idx]
        pred_occluded[0, track_idx] = ~pred_visible[:, track_idx].squeeze()

        gt_points[0, track_idx] = gt_track_points
        pred_points[0, track_idx] = pred_trajs_norm[:, track_idx, :]

    gt_points *= scale
    pred_points *= scale

    metrics = compute_tapvid_metrics(query_points, gt_occluded,
                                     gt_points, pred_occluded,
                                     pred_points, 'first')

    for key in metrics:
        metrics[key] = metrics[key] * 100
    return metrics
