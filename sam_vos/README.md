```md
# SAM-2 Interactive Video Object Segmentation

This project provides a lightweight GUI to interactively prompt an object on the first frame using points and automatically propagate the segmentation across all frames using **SAM-2**.

---

## Folder Structure
```

sam_vos/
├── app_gui.py
├── app_core.py
├── engine.py
├── point_manager.py
└── data/
├── images/ # Input frames (ordered images)
└── vos_results/ # Output segmentation overlays

````

---

## How to Run

Activate your Python environment and run:

```bash
python app_gui.py --image_dir data/images
````

---

## GUI Interaction

### Mouse Controls

- **Left Click** → Foreground point (green)
- **Right Click** → Background point (red)
- **Ctrl + Left Click** → Background point (macOS support)

### Buttons

- **Clear**
  Clears all placed points so you can re-annotate.

- **Visualization**
  Shows how SAM-2 interprets the current points on the first frame only.
  You can add more points and visualize again for refinement.

- **Segment & Run**
  Finalizes the object, closes the GUI, and automatically propagates the
  segmentation to all frames.

---

## Output

Segmented results are saved to:

```
data/vos_results/
```

One output image is generated per frame (`frame_0.png`, `frame_1.png`, …).

---

## Notes

- Works on **Windows, macOS, and Linux**
- CUDA is used automatically if available; otherwise, CPU is used
- On macOS, use **Ctrl + Left Click** for background points
