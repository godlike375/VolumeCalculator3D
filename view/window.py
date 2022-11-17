from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from numpy import ndarray
from view.pyplot_qt import MplCanvas


class MainForm(QMainWindow):
    def __init__(self, view_model):
        super().__init__()
        self._view_model = view_model
        self.setWindowTitle("3D volume calculator")
        layout = QVBoxLayout()
        self.chose_folder = QPushButton("Выбрать папку с фото")
        self.chose_folder.clicked.connect(self.select)
        layout.addWidget(self.chose_folder)
        self.plot = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.plot)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.Directory)

        if dlg.exec():
            self._folder = dlg.selectedFiles()

    def show_3d_object(self, x: ndarray, y:ndarray, z:ndarray):
        #zdata = 20 * np.random.random(30)
        #xdata = np.sin(zdata) + 0.3 * np.random.randn(30)
        #ydata = np.cos(zdata) + 0.3 * np.random.randn(30)
        self.plot.axes.scatter3D(x, y, z)