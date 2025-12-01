import cv2

class PointManager:
    def __init__(self):
        self.points = []
        self.labels = []   # 1 = positive, 0 = negative

    def add_point(self, x, y, label):
        self.points.append([x, y])
        self.labels.append(label)

    def clear(self):
        self.points = []
        self.labels = []


def mouse_handler(event, x, y, flags, point_manager):
    if event == cv2.EVENT_LBUTTONDOWN:     # Positive point
        point_manager.add_point(x, y, 1)

    elif event == cv2.EVENT_RBUTTONDOWN:   # Negative point
        point_manager.add_point(x, y, 0)
