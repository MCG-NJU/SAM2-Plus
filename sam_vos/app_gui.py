import os
import cv2
import torch
import argparse
import tkinter as tk
from PIL import Image, ImageTk

from sam2.build_sam import build_sam2_video_predictor
from point_manager import PointManager
from app_core import AppCore
from engine import overlay_mask
import warnings

warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'"
)

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True, type=str)
args = parser.parse_args()

os.environ["TQDM_DISABLE"] = "1"

IMAGE_DIR = args.image_dir

MODEL_CFG = r"F:/GitHub/SAM2-Plus/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
MODEL_CKPT = r"F:/GitHub/SAM2-Plus/checkpoints/sam2.1_hiera_tiny.pt"

DISPLAY_W, DISPLAY_H = 900, 500


# ---------------- LOAD FRAMES ----------------
frames = sorted(
    [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
     if f.lower().endswith((".jpg", ".png"))]
)
assert len(frames) > 0, "No images found"


# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = build_sam2_video_predictor(
    MODEL_CFG,
    MODEL_CKPT,
    device=device,
)

print("[OK] SAM-2 model and dependencies loaded successfully.")

state = predictor.init_state(IMAGE_DIR)


# ---------------- CORE ----------------
pm = PointManager()
core = AppCore(predictor, frames, state, pm)

preview_mask = None   # for Visualization button


# ---------------- GUI ----------------
root = tk.Tk()
root.title("SAM-2 Prompt (Once)")

canvas = tk.Canvas(root, width=DISPLAY_W, height=DISPLAY_H, bg="black")
canvas.pack()

tk_img = None


def draw(img):
    global tk_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
    tk_img = ImageTk.PhotoImage(Image.fromarray(img))
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)


def draw_points(img):
    for (x, y), lbl in zip(pm.points, pm.labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(img, (x, y), 6, color, -1)


def update():
    img = cv2.imread(frames[0])

    if preview_mask is not None:
        img = overlay_mask(img, preview_mask)

    draw_points(img)
    draw(img)


# ---------------- MOUSE ----------------
def on_click(event):
    img = cv2.imread(frames[0])
    h, w = img.shape[:2]

    x = int(event.x * w / DISPLAY_W)
    y = int(event.y * h / DISPLAY_H)

    if event.num == 1:
        pm.add_point(x, y, 1)   # FG
    elif event.num == 3:
        pm.add_point(x, y, 0)   # BG

    update()


canvas.bind("<Button-1>", on_click)
canvas.bind("<Button-3>", on_click)


# ---------------- BUTTON ACTIONS ----------------
def clear_points():
    global preview_mask
    pm.clear()
    preview_mask = None
    update()


def visualize_prompt():
    """
    Visualize SAM-2 response on frame 0 ONLY.
    Does NOT commit object memory.
    """
    global preview_mask

    if len(pm.points) == 0:
        return

    res = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        points=pm.points,
        labels=pm.labels,
        obj_id=0,
    )

    logits = res[2][0].squeeze().cpu()
    preview_mask = (torch.sigmoid(logits) > 0.5).numpy().astype("uint8")

    update()


def segment_and_run():
    core.initialize_object()
    root.destroy()
    core.run_full_propagation()


# ---------------- BUTTONS ----------------
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Clear", width=12, command=clear_points).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Visualization", width=12, command=visualize_prompt).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Segment & Run", width=14, command=segment_and_run).grid(row=0, column=2, padx=5)


# ---------------- START ----------------
update()
root.mainloop()
