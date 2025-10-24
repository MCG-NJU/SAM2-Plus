import os
import numpy as np
import torch
from tqdm import tqdm
from .metrics import calc_seq_err_robust, get_auc_curve, get_prec_curve

def extract_results_helper(track_result_dir,
                    dataset,
                    skip_missing_seq=False,
                    plot_bin_gap=0.05,
                    exclude_invalid_frames=False):
    """
    Extract results for the given directory
    @param
        track_result_dir: directory containing the tracker results
        dataset: The dataset object.
        skip_missing_seq: If True, skip missing sequences. Default is False
        plot_bin_gap: gap between thresholds for overlap plot
        exclude_invalid_frames: If True, exclude invalid frames from the evaluation. Default is False
    @return:
        eval_data: Dictionary containing the extracted results
    """

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros(len(dataset), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((len(dataset), threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((len(dataset), threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((len(dataset), threshold_set_center.numel()),
                                                    dtype=torch.float32)
    
    valid_sequence = torch.ones(len(dataset), dtype=torch.uint8)

    for seq_id, seq in enumerate(tqdm(dataset.get_sequence())):
        # Load anno
        anno_bb = torch.tensor(seq.ground_truth_rect)
        target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None

        # Load results
        results_path = os.path.join(track_result_dir, seq.get_name() + '.txt')
        if not os.path.isfile(results_path) and "/" in seq.get_name(): # for lasot dataset
            results_path = os.path.join(track_result_dir, seq.get_name().split("/")[-1] + '.txt')

        if os.path.isfile(results_path):
            pred_bb = np.loadtxt(results_path, dtype=int)
            pred_bb = torch.tensor(pred_bb)
        else:
            if skip_missing_seq:
                valid_sequence[seq_id] = 0
                break
            else:
                raise Exception('Result not found. {}'.format(results_path))

        # Calculate measures
        err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
            pred_bb, anno_bb, seq.dataset, target_visible)

        avg_overlap_all[seq_id] = err_overlap[valid_frame].mean()

        if exclude_invalid_frames:
            seq_length = valid_frame.long().sum()
        else:
            seq_length = anno_bb.shape[0]

        if seq_length <= 0:
            raise Exception('Seq length zero')

        ave_success_rate_plot_overlap[seq_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
        ave_success_rate_plot_center[seq_id :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
        ave_success_rate_plot_center_norm[seq_id :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))


    eval_data = {'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    return eval_data


def get_and_save_results(track_result_dir, 
                         dataset, 
                         result_file_path, 
                         plot_types=('success', 'prec', 'norm_prec'), 
                         ):
    """ Print the results for the given trackers in a formatted table
    @param:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        result_file_path - path of the result_file (should be a .txt file)
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
    @return:
        scores - Dictionary containing the computed scores['AUC', 'OP50', 'OP75', 'Precision', 'Norm Precision']
    """

    eval_data = extract_results_helper(track_result_dir, dataset)

    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    
    # save result to file
    with open (result_file_path, 'w') as f:
        for key in scores.keys():
            f.write(f"{key}: {scores[key].item()}\n")

    return scores

