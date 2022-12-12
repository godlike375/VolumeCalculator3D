from PyQt6.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLabel,
    QTextEdit, QHBoxLayout
)
from matplotlib import pyplot as plt
from numpy import ndarray

from view.pyplot_qt import Plot3D
from view.view_model import VOLUME

from model.business_logic import Model
from view.view_model import ViewModel

from common.logger import logger

class MainForm(QMainWindow):
    def __init__(self, view_model):
        super().__init__()
        self._view_model = view_model
        self.setWindowTitle('3D volume calculator')
        layout = QVBoxLayout()

        self.chose_folder = QPushButton('Выбрать папку с изображениями')
        self.chose_folder.clicked.connect(self.clear_plot_and_volume)
        self.chose_folder.clicked.connect(self.select)

        horiz_layout = QHBoxLayout()
        layout.addLayout(horiz_layout)

        self.volume = QLabel()
        self.set_volume(0)
        horiz_layout.addWidget(self.chose_folder)


        self.approximation = QLabel('Коэффициент аппроксимации = ')
        horiz_layout.addWidget(self.approximation)

        self.approximation_rate = QTextEdit(f'{Model.DEFAULT_APPROXIMATION_RATE}')
        self.approximation_rate.setFixedHeight(25)
        self.approximation_rate.textChanged.connect(self.approximation_rate_changed)
        horiz_layout.addWidget(self.approximation_rate)
        horiz_layout.addWidget(self.volume)
        self.plot = Plot3D(self, width=15, height=15, dpi=160)
        layout.addWidget(self.plot)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.Directory)

        if dlg.exec():
            folder_dir = dlg.selectedFiles()[0]
            logger.debug('selected files')
            self._view_model.model_run(folder_dir)
        else:
            ViewModel.show_message('Ошибка', 'Необходимо выбрать папку')

    def approximation_rate_changed(self):
        rate = self.approximation_rate.toPlainText()
        floated = float(rate)
        self._view_model.set_approximation_rate(floated)

    def clear_plot_and_volume(self):
        self.plot.axes.cla()
        self.set_volume(0)
        self.plot.clear_axes_labels()

    def draw_point_cloud(self, xs: ndarray, ys: ndarray, zs: ndarray):
        colormap = plt.get_cmap("turbo")
        self.plot.axes.scatter3D(xs, ys, zs, s=1, c=zs, cmap=colormap)
        self.plot.clear_axes_labels()

    def set_volume(self, volume: float):
        self.volume.setText(f'{VOLUME} {volume} мл')
