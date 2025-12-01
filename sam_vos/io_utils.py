import os
import cv2

def load_frames(video_dir):
    """
    Reads frames from a directory in numeric order.
    """
    frames = sorted(os.listdir(video_dir), key=lambda x: int(os.path.splitext(x)[0]))
    frames = [os.path.join(video_dir, f) for f in frames]
    return frames


def save_output(image, out_path):
    cv2.imwrite(out_path, image)
