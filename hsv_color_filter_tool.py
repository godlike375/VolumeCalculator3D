import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import traceback

# Константы для имен окон и трекбаров
WINDOW_NAME = 'HSV Color Filter'

TRACKBAR_H_MIN = 'H_min'
TRACKBAR_S_MIN = 'S_min'
TRACKBAR_V_MIN = 'V_min'
TRACKBAR_H_MAX = 'H_max'
TRACKBAR_S_MAX = 'S_max'
TRACKBAR_V_MAX = 'V_max'

# Диапазоны HSV
HUE_MAX = 179
SAT_MAX = 255
VAL_MAX = 255

TRACKBAR_CONTROLS_WINDOW_NAME = 'Trackbar Controls'

def nothing(x):
    """
    Пустая функция обратного вызова для трекбаров.
    cv2.createTrackbar требует функцию обратного вызова, но нам не нужна
    немедленная реакция, так как значения считываются в основном цикле.
    """
    pass

def create_trackbars():
    """
    Создает ползунки для настройки HSV диапазонов.
    """
    cv2.createTrackbar(TRACKBAR_H_MIN, TRACKBAR_CONTROLS_WINDOW_NAME, 0, HUE_MAX, nothing)
    cv2.createTrackbar(TRACKBAR_S_MIN, TRACKBAR_CONTROLS_WINDOW_NAME, 0, SAT_MAX, nothing)
    cv2.createTrackbar(TRACKBAR_V_MIN, TRACKBAR_CONTROLS_WINDOW_NAME, 0, VAL_MAX, nothing)
    cv2.createTrackbar(TRACKBAR_H_MAX, TRACKBAR_CONTROLS_WINDOW_NAME, HUE_MAX, HUE_MAX, nothing) # Изначально верхние границы установлены в максимум
    cv2.createTrackbar(TRACKBAR_S_MAX, TRACKBAR_CONTROLS_WINDOW_NAME, SAT_MAX, SAT_MAX, nothing)
    cv2.createTrackbar(TRACKBAR_V_MAX, TRACKBAR_CONTROLS_WINDOW_NAME, VAL_MAX, VAL_MAX, nothing)

def get_trackbar_values():
    """
    Считывает текущие значения со всех ползунков.
    """
    h_min = cv2.getTrackbarPos(TRACKBAR_H_MIN, TRACKBAR_CONTROLS_WINDOW_NAME)
    s_min = cv2.getTrackbarPos(TRACKBAR_S_MIN, TRACKBAR_CONTROLS_WINDOW_NAME)
    v_min = cv2.getTrackbarPos(TRACKBAR_V_MIN, TRACKBAR_CONTROLS_WINDOW_NAME)
    h_max = cv2.getTrackbarPos(TRACKBAR_H_MAX, TRACKBAR_CONTROLS_WINDOW_NAME)
    s_max = cv2.getTrackbarPos(TRACKBAR_S_MAX, TRACKBAR_CONTROLS_WINDOW_NAME)
    v_max = cv2.getTrackbarPos(TRACKBAR_V_MAX, TRACKBAR_CONTROLS_WINDOW_NAME)
    return h_min, s_min, v_min, h_max, s_max, v_max

def _imread_unicode(path):
    """
    Чтение изображения с поддержкой Unicode путей
    """
    try:
        with open(path, "rb") as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось декодировать изображение: {path}")
        return img
    except Exception as e:
        print(f"Ошибка чтения изображения {path}: {str(e)}")
        traceback.print_exc()
        return None

def select_image_file():
    """
    Открывает файловый диалог для выбора изображения.
    """
    root = tk.Tk()
    root.withdraw()  # Скрыть основное окно tkinter
    
    # Открываем диалог выбора файла
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("Все файлы", "*.*")
        ]
    )
    
    root.destroy()  # Закрыть tkinter окно
    
    return file_path

def show_error(message, exc=None, tb=None):
    """
    Отображение ошибок (замените на вашу реализацию)
    """
    print(f"ОШИБКА: {message}")
    if exc:
        print(f"Исключение: {exc}")
    if tb:
        print(f"Стек вызовов:\n{tb}")

def main():
    # Открываем файловый диалог для выбора изображения
    image_path = select_image_file()
    
    # Проверяем, был ли выбран файл
    if not image_path or not os.path.exists(image_path):
        print("Ошибка: Файл изображения не выбран или не найден")
        print("Программа завершена.")
        return

    # Используем вашу функцию для чтения изображения
    image = _imread_unicode(image_path)
    
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение по пути: {image_path}")
        print("Возможно, файл поврежден или имеет неподдерживаемый формат.")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(TRACKBAR_CONTROLS_WINDOW_NAME) # Окно для трекбаров
    cv2.resizeWindow(TRACKBAR_CONTROLS_WINDOW_NAME, 250, 400) # Set size for vertical stacking of trackbars

    create_trackbars()

    while True:
        # Получаем текущие значения HSV с ползунков
        h_min, s_min, v_min, h_max, s_max, v_max = get_trackbar_values()

        # Создаем массивы нижних и верхних границ HSV
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Преобразуем изображение из BGR в HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Применяем фильтр InRange для создания маски
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Применяем маску к оригинальному изображению
        # Это показывает только те части оригинального изображения, которые попадают в HSV диапазон
        result = cv2.bitwise_and(image, image, mask=mask)

        # Отображаем только результат фильтрации
        result = cv2.resize(result, (900, 600))
        cv2.imshow(WINDOW_NAME, result)

        # Ждем 1 миллисекунду, проверяем нажатие клавиши ESC (ASCII 27) для выхода
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # Закрываем все окна OpenCV при выходе
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()