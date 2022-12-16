from configparser import ConfigParser

from pathlib import Path

from common.logger import logger

FOLDER = 'config'
FILE = 'settings.ini'
AREA_FILE = 'selected_area.pickle'
ROOT_FOLDER = 'VolumeCalculator'




def get_repo_path(current: Path = None):
    current_path = current or Path.cwd()
    while current_path.name != ROOT_FOLDER:
        if current_path == current_path.parent:
            logger.exception(f'Корневая директория программы "{ROOT_FOLDER}" не найдена')
            return Path.cwd()
        current_path = current_path.parent
    return current_path


class Settings:
    DEFAULT_APPROXIMATION_RATE = 0.0013

    @classmethod
    def load(cls, folder: str = FOLDER, file: str = FILE):
        base_path = get_repo_path()

        path = base_path / folder / file
        if Path.exists(path):
            config = ConfigParser()
            config.read(str(path))
            for sec in config.sections():
                for key, value in config[sec].items():
                    setattr(cls, key.upper(), float(value) if '.' in value else int(value))

    @classmethod
    def save(cls, folder: str = FOLDER, file: str = FILE):
        base_path = get_repo_path()

        config = ConfigParser()
        fields = {k: vars(cls)[k] for k in vars(cls) if k.isupper()}
        config['settings'] = fields
        path = base_path / folder
        Path.mkdir(path, exist_ok=True)
        with open(path / file, 'w') as file:
            config.write(file)

    @classmethod
    def set_approximation_rate(cls, rate):
        cls.DEFAULT_APPROXIMATION_RATE = rate
