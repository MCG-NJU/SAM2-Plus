import tkinter as tk
from tkinter import filedialog, messagebox
import cv2, os
from PIL import Image, ImageTk

from app_core import VOSCore
from engine import overlay_mask
import numpy as np

core=VOSCore()
SAVE_DIR="vos_results/"

MODEL_CFG  = r"F:\GitHub\SAM2-Plus\sam2\configs\sam2.1\sam2.1_hiera_t.yaml"
MODEL_CKPT = r"F:\GitHub\SAM2-Plus\checkpoints\sam2.1_hiera_tiny.pt"

DISPLAY_W=900
DISPLAY_H=500


def update_frame():
    if not core.frames:
        canvas.after(30, update_frame)
        return

    img = core.current_frame().copy()

    # overlay mask
    if isinstance(core.mask, np.ndarray):
        img = overlay_mask(img, core.mask)


    disp = cv2.resize(img,(DISPLAY_W,DISPLAY_H))
    disp = cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)

    for (px,py),lbl in zip(core.pm.points,core.pm.labels):
        sx = int(px * DISPLAY_W / img.shape[1])
        #sy = int(py * DISPLAY_HEIGHT / img.shape[0])
        sy = int(py * DISPLAY_H / img.shape[0])
        cv2.circle(disp,(sx,sy),7,(0,255,0) if lbl==1 else (255,0,0),-1)

    imgtk = ImageTk.PhotoImage(Image.fromarray(disp))
    canvas.img = imgtk
    canvas.create_image(0,0,anchor=tk.NW,image=imgtk)

    canvas.after(30, update_frame)


def click_fg(e):
    img=core.current_frame()
    H,W=img.shape[:2]
    px=int(e.x*W/DISPLAY_W)
    py=int(e.y*H/DISPLAY_H)
    core.pm.add_point(px,py,1)
    print("[CLICK+FG]",px,py)

def click_bg(e):
    img=core.current_frame()
    H,W=img.shape[:2]
    px=int(e.x*W/DISPLAY_W)
    py=int(e.y*H/DISPLAY_H)
    core.pm.add_point(px,py,0)
    print("[CLICK-BG]",px,py)


def load_model():
    core.load_model(MODEL_CFG,MODEL_CKPT)

def load_frames():
    p=filedialog.askdirectory()
    if p:
        core.load_video(p)
        update_frame()

def segment():
    print("[UI] Segment clicked")
    core.segment()

def nxt(): core.next_frame()
def prv(): core.prev_frame()
def clr(): core.pm.clear();core.mask=None;print("[UI] points cleared")

def save_one():
    img=core.current_frame().copy()
    if core.mask is not None: img=overlay_mask(img,core.mask)
    os.makedirs(SAVE_DIR,exist_ok=True)
    fp=f"{SAVE_DIR}/frame_{core.frame_idx}.png"
    cv2.imwrite(fp,img)
    print("[SAVE]",fp)

def run_all():
    print("[RUN FULL]")
    if not core.pm.points: return print("❌ No points — abort")

    os.makedirs(SAVE_DIR,exist_ok=True)

    for i in range(core.frame_idx,len(core.frames)):
        print(f"[FULL] frame {i}")
        core.segment()
        img=overlay_mask(core.current_frame(),core.mask)
        cv2.imwrite(f"{SAVE_DIR}/frame_{i}.png",img)
        core.next_frame()


root=tk.Tk()
root.title("SAM2 VOS — FULL LOG MODE")

canvas=tk.Canvas(root,width=DISPLAY_W,height=DISPLAY_H,bg="black")
canvas.pack()
canvas.bind("<Button-1>",click_fg)
canvas.bind("<Button-3>",click_bg)

for t,c in [
    ("Load Model",load_model),
    ("Select Frames Folder",load_frames),
    ("Segment Frame",segment),
    ("← Prev",prv),
    ("Next →",nxt),
    ("Clear Points",clr),
    ("Save Current Frame",save_one),
    ("Run Full Sequence",run_all)
]:
    tk.Button(root,text=t,command=c).pack(fill="x")

root.mainloop()
