import sys

from PyQt6.QtWidgets import QApplication

from model.business_logic import Model
from view.view_model import ViewModel
from view.window import MainForm
from common.logger import logger


def main():
    logger.debug('program started')
    try:
        view_model = ViewModel()
        model = Model(view_model)
        app = QApplication(sys.argv)
        view = MainForm(view_model)
        view_model.set_model(model)
        view_model.set_view(view)
        view.show()
        app.exec()
    except Exception as e:
        ViewModel.show_message('Непредвиденная ошибка', str(e))
        logger.critical(str(e))


if __name__ == '__main__':
    main()

# TODO: сделать распознавание цифр или линий
