import numpy
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLabel, QMessageBox
)
from numpy import ndarray

from view.pyplot_qt import MplCanvas

from view.view_model import VOLUME

class MainForm(QMainWindow):
    def __init__(self, view_model):
        super().__init__()
        self._view_model = view_model
        self.setWindowTitle('3D volume calculator')
        layout = QVBoxLayout()

        self.chose_folder = QPushButton('Выбрать папку со снимками')
        self.chose_folder.clicked.connect(self.select)
        layout.addWidget(self.chose_folder)

        self.volume = QLabel(VOLUME)
        layout.addWidget(self.volume)

        self.plot = MplCanvas(self, width=15, height=15, dpi=150)
        layout.addWidget(self.plot)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.Directory)

        if dlg.exec():
            self._view_model.model_run(dlg.selectedFiles()[0])
        else:
            message = QMessageBox()
            message.setText('Необходимо выбрать файл')
            message.setWindowTitle('Ошибка')
            message.exec()

    def show_3d_object(self, x: ndarray, z:ndarray, y:ndarray):
        #zdata = 20 * np.random.random(30)
        #xdata = np.sin(zdata) + 0.3 * np.random.randn(30)
        #ydata = np.cos(zdata) + 0.3 * np.random.randn(30)
        cmhot = plt.get_cmap("turbo")
        s = numpy.ones(len(x))
        self.plot.axes.scatter3D(x, y, z, s=s, c=z, cmap=cmhot)