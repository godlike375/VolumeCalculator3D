from PyQt6.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLabel, QMessageBox
)
from matplotlib import pyplot as plt
from numpy import ndarray

from view.pyplot_qt import Plot3D
from view.view_model import VOLUME


class MainForm(QMainWindow):
    def __init__(self, view_model):
        super().__init__()
        self._view_model = view_model
        self.setWindowTitle('3D volume calculator')
        layout = QVBoxLayout()

        self.chose_folder = QPushButton('Выбрать папку со снимками')
        self.chose_folder.clicked.connect(self.clear_plot_and_volume)
        self.chose_folder.clicked.connect(self.select)
        layout.addWidget(self.chose_folder)

        self.volume = QLabel()
        self.set_volume(0)
        layout.addWidget(self.volume)

        self.plot = Plot3D(self, width=15, height=15, dpi=150)
        layout.addWidget(self.plot)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.Directory)

        if dlg.exec():
            folder_dir = dlg.selectedFiles()[0]
            self._view_model.model_run(folder_dir)
        else:
            self.show_message('Ошибка', 'Необходимо выбрать папку')

    def clear_plot_and_volume(self):
        self.plot.axes.cla()
        self.set_volume(0)

    def draw_point_cloud(self, xs: ndarray, ys: ndarray, zs: ndarray):
        colormap = plt.get_cmap("turbo")
        self.plot.axes.scatter3D(xs, ys, zs, s=1, c=zs, cmap=colormap)

    def set_volume(self, volume: float):
        self.volume.setText(f'{VOLUME} {volume}')

    def show_message(self, title, text):
        message = QMessageBox()
        message.setText(text)
        message.setWindowTitle(title)
        message.exec()
