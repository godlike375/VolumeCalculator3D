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