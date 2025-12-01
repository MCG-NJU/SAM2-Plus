from model import load_sam2, initialize_inference
from io_utils import load_frames, save_output
from interaction import PointManager, mouse_handler
from engine import run_segmentation, overlay_mask

import cv2, os

# # TODO — update with real paths before running
# MODEL_CFG  = "sam2.1_hiera_t.yaml"
# MODEL_CKPT = "sam2.1_hiera_tiny.pt"
# VIDEO_DIR  = r"frames/"           # directory of numbered frames
SAVE_DIR   = "results/"           # results output folder

MODEL_CFG = r"F:\GitHub\SAM2-Plus\sam2\configs\sam2.1\sam2.1_hiera_t.yaml"
MODEL_CKPT = r"F:\GitHub\SAM2-Plus\checkpoints\sam2.1_hiera_tiny.pt"
VIDEO_DIR = r"F:\GitHub\SAM2-Plus\notebook\data\images"



def draw_status(image, text, y=30, color=(255,255,255)):
    """Render UI hint text on video frame."""
    cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():

    predictor, device = load_sam2(MODEL_CFG, MODEL_CKPT)
    state = initialize_inference(predictor, VIDEO_DIR)
    frames = load_frames(VIDEO_DIR)

    os.makedirs(SAVE_DIR, exist_ok=True)

    pm = PointManager()
    cv2.namedWindow("SAM-VOS")
    cv2.setMouseCallback("SAM-VOS", lambda e,x,y,f: mouse_handler(e,x,y,f, pm))

    frame_idx = 0
    mask = None

    while True:

        img = cv2.imread(frames[frame_idx])
        vis = img.copy()

        # Overlay last mask if still active
        if mask is not None:
            vis = overlay_mask(vis, mask)

        # UI instructions on screen
        draw_status(vis, f"Frame: {frame_idx+1}/{len(frames)}", 30)
        draw_status(vis, "L-click = + point | R-click = - point", 60)
        draw_status(vis, "S = Segment | C = Clear | N=next | B=back", 90)
        draw_status(vis, "W = Save mask | Q = Quit", 120)

        cv2.imshow("SAM-VOS", vis)
        key = cv2.waitKey(1) & 0xFF

        # ---- Actions ----

        if key == ord('s'):                 # run segmentation
            mask = run_segmentation(predictor, state, frame_idx, pm.points, pm.labels)

        elif key == ord('c'):               # clear prompt only
            pm.clear()
            mask = None

        elif key == ord('n'):               # next frame
            frame_idx = min(frame_idx+1, len(frames)-1)
            pm.clear()
            mask = None

        elif key == ord('b'):               # previous frame
            frame_idx = max(frame_idx-1, 0)
            pm.clear()
            mask = None

        elif key == ord('w'):               # save output
            if mask is not None:
                out_path = os.path.join(SAVE_DIR, f"frame_{frame_idx}_mask.png")
                save_output(vis, out_path)
                print(f"[✔] Saved: {out_path}")

        elif key == ord('q'):               # quit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
