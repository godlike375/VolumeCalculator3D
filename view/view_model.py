from PyQt6.QtWidgets import QMessageBox

VOLUME = 'Объём лекарства ='


class ViewModel:
    def __init__(self):
        self._model = None
        self._view = None

    def set_model(self, model):
        self._model = model

    def set_view(self, view):
        self._view = view

    def model_run(self, dir: str):
        self._model.get_volume_by_images(dir)

    def set_volume(self, volume):
        self._view.set_volume(volume)

    def set_points(self, points):
        self._view.draw_point_cloud(*points)

    @staticmethod
    def show_message(title, text):
        message = QMessageBox()
        message.setText(text)
        message.setWindowTitle(title)
        message.exec()

    def set_approximation_rate(self, rate):
        self._model.set_approximation_rate(rate)
