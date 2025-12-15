class PointManager:
    """
    Stores user interaction points for SAM-2.
    labels:
        1 -> positive (foreground)
        0 -> negative (background)
    """

    def __init__(self):
        self.points = []
        self.labels = []

    def add_point(self, x, y, label):
        self.points.append([x, y])
        self.labels.append(label)

    def clear(self):
        self.points = []
        self.labels = []
