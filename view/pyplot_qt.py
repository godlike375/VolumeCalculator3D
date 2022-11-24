from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Plot3D(FigureCanvasQTAgg):

    def __init__(self, parent, width, height, dpi):
        figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = figure.add_subplot(111, projection='3d')
        super().__init__(figure)
