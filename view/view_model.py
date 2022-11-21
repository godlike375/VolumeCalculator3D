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
        self._view.set_volume(volume)

    def set_points(self, points):
        res = points.tolist()
        self._view.show_3d_object(*res)

    def show_message(self, title, message):
        self._view.show_message(title, message)