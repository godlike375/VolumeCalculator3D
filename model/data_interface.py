from pathlib import Path
from dataclasses import dataclass

from numpy import ndarray
import cv2
import re
import os

IMAGE_NUMBER = r'_(\d+)\.'

@dataclass
class NumberedImage:
    image: ndarray
    number: int

def get_files_with_numbers(dir):
    images = []
    old_dir = Path.cwd()
    os.chdir(dir)
    for file in Path.iterdir(Path(dir)):
        matched = re.search(IMAGE_NUMBER, file.name)
        if matched:
            number = int(matched.group(1))
            image = cv2.imread(file.name)
            images.append(NumberedImage(image, number))
    os.chdir(old_dir)
    return images