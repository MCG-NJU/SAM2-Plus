import numpy as np
import pandas as pd
from PIL import Image
import cv2


def imread_indexed(filename):
    """ Load indexed image with given filename. Used to read segmentation annotations."""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation



def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)



def _draw_sequence(frames_path, frames_name_list, ground_truth_rect, save_dir):
    """
    Draws the sequence with the ground truth bounding box.
    Args:
        frames_path (str): The path to the frames.
        frames_name_list (list): The list of frame names.
        ground_truth_rect (np.ndarray): The ground truth bounding box.
        save_dir (str): The directory to save the drawn sequence.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Drawing sequence to {}'.format(save_dir))
    for i, frame_name in enumerate(frames_name_list):
        frame_path = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_path)
        x, y, w, h = ground_truth_rect[i]
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        save_path = os.path.join(save_dir, frame_name)
        cv2.imwrite(save_path, frame)