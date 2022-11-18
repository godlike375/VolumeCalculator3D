VOLUME = 'Объём ='
class ViewModel:
    def __init__(self):
        self._model = None
        self._view = None

    def set_model(self, model):
        self._model = model

    def set_view(self, view):
        self._view = view

    def model_run(self, dir: str):
        self._model.run(dir)

    def set_volume(self, volume):
        self._view.volume.setText(f'{VOLUME} {volume}')

    def set_points(self, points):
        self._view.show_3d_object(points)