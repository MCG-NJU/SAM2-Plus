import os
import cv2
import torch
from engine import initialize_object, overlay_mask


class AppCore:
    def __init__(self, predictor, frames, state, point_manager):
        self.predictor = predictor
        self.frames = frames
        self.state = state
        self.pm = point_manager
        self.obj_id = 0

    def initialize_object(self):
        """
        Initialize SAM-2 object memory using user prompts on frame 0.
        """
        print("[INFO] Initializing object with user prompt...")

        initialize_object(
            self.predictor,
            self.state,
            frame_idx=0,
            points=self.pm.points,
            labels=self.pm.labels,
            obj_id=self.obj_id,
        )

        print("[OK] Object initialized.")

    def run_full_propagation(self):
        """
        Run SAM-2 propagation on all remaining frames and save results.
        Output directory: ../vos_results relative to image directory.
        """
        print("[INFO] Running SAM-2 propagation...")

        # ------------------------------------------------
        # Output directory: data/vos_results
        # ------------------------------------------------
        images_dir = os.path.dirname(self.frames[0])      # .../data/images
        output_dir = os.path.abspath(
            os.path.join(images_dir, "..", "vos_results")
        )
        os.makedirs(output_dir, exist_ok=True)

        # ------------------------------------------------
        # SAM-2 propagation (generator – must be called ONCE)
        # ------------------------------------------------
        for frame_idx, obj_ids, logits, _, _ in self.predictor.propagate_in_video(self.state):

            # logits shape: (num_objects, 1, H, W)
            logits = logits[0].squeeze().cpu()
            mask = (torch.sigmoid(logits) > 0.5).numpy().astype("uint8")

            img = cv2.imread(self.frames[frame_idx])
            vis = overlay_mask(img, mask)

            out_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
            cv2.imwrite(out_path, vis)

        print(f"[OK] Propagation completed. Results saved to:\n{output_dir}")
