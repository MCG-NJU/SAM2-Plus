# SAM-2 Interactive Video Object Segmentation

This project provides a lightweight GUI to interactively prompt an object on the first frame using points and automatically propagate the segmentation across all frames using **SAM-2**.

## Folder Structure

sam_vos/

- app_core.py
- app_gui.py
- engine.py
- point_manager.py
- data/
  - images # Input frames
  - vos_results # Output segmentation overlays

## How to Run

1. Activate your Python environment
2. `cd sam_vos`
3. `pip install -r requirements.txt`
4. `python app_gui.py --image_dir data/images`

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

## Output

Segmented results are saved to:

```
data/vos_results/
```

## Notes

- Works on **Windows, macOS, and Linux**
- CUDA is used automatically if available; otherwise, CPU is used
- On macOS, use **Ctrl + Left Click** for background points
