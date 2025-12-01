from model import load_sam2, initialize_inference
from engine import run_segmentation
import cv2


class VOSCore:
    def __init__(self):
        self.predictor=None
        self.state=None
        self.frames=[]
        self.frame_idx=0
        self.pm=None

        from interaction import PointManager
        self.pm = PointManager()
        self.mask=None

    def load_model(self,cfg,ckpt):
        print("[LOG] Loading model...")
        self.predictor,_ = load_sam2(cfg,ckpt)
        print("[OK] Model loaded.")

    def load_video(self,video_dir):
        from io_utils import load_frames
        print("[LOG] Loading frames from:",video_dir)
        self.frames = load_frames(video_dir)
        print("[OK] Frames loaded:",len(self.frames))

        print("[LOG] Initializing VOS state...")
        self.state = initialize_inference(self.predictor,video_dir)
        self.frame_idx=0
        self.mask=None
        print("[OK] State ready at frame 0")

    def current_frame(self):
        return cv2.imread(self.frames[self.frame_idx])

    def segment(self):
        if not self.pm.points: return

        from engine import run_segmentation
        self.mask = run_segmentation(
            predictor=self.predictor,
            state=self.state,
            frame=self.frame_idx,
            points=self.pm.points,
            labels=self.pm.labels
        )

    def next_frame(self):
        if self.frame_idx < len(self.frames)-1:
            self.frame_idx+=1
            self.mask=None
        print("[MOVE] → frame",self.frame_idx)

    def prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx-=1
            self.mask=None
        print("[MOVE] ← frame",self.frame_idx)

