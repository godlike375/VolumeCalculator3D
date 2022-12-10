from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Plot3D(FigureCanvasQTAgg):

    def __init__(self, parent, width, height, dpi):
        figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = figure.add_subplot(111, projection='3d')
        self.clear_axes_labels()
        super().__init__(figure)

    def clear_axes_labels(self):
        self.axes.xaxis.set_ticklabels([])
        self.axes.yaxis.set_ticklabels([])
        self.axes.zaxis.set_ticklabels([])
