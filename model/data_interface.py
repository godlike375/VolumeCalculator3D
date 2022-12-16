from pathlib import Path
from dataclasses import dataclass
import re
import os

from numpy import ndarray
import cv2

from common.logger import logger

IMAGE_NUMBER = r'_(\d+)\.'


@dataclass
class NumberedImage:
    image: ndarray
    number: int


def get_files_with_numbers(dir):
    images = []
    old_dir = Path.cwd()
    os.chdir(dir)
    logger.debug(f'chosen directory: {dir}')
    logger.debug('filenames:')
    for file in Path.iterdir(Path(dir)):
        image, number = read_image(file.name)
        if image is not None:
            logger.debug(f'{file.name}, number: {number}')
            images.append(NumberedImage(image, number))
    os.chdir(old_dir)
    return images

def read_image(filename):
    number = check_image_number(filename)
    if number is not None:
        return cv2.imread(filename), number
    return None, None

def check_image_number(filename):
    matched = re.search(IMAGE_NUMBER, filename)
    if matched:
        number = int(matched.group(1))
        return number