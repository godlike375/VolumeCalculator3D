import logging
from pathlib import Path
import dataclasses
from typing import List

import cv2
import numpy as np
import pyvista as pv
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction, QImage, QPixmap, QPen, QBrush, QColor
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsEllipseItem
from PyQt6.QtCore import Qt, QPointF, QRectF
from pyvistaqt import QtInteractor

# === КОНСТАНТЫ ЛОГИРОВАНИЯ ===
LOG_FILENAME = 'scan_processor.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

# === КОНСТАНТЫ ОБРАБОТКИ ИЗОБРАЖЕНИЙ ===
ROI_PERCENTAGE = 0.025
MIN_CONTOUR_AREA = 4
CONFIDENCE_THRESHOLD = 0.7
TARGET_NORM_SIZE = (20, 32)  # (ширина, высота)
MORPH_KERNEL_MAX_SIZE = 1
TEMPLATES_DIR = "templates"

# === КОНСТАНТЫ МОДЕЛИРОВАНИЯ И ВЫЧИСЛЕНИЯ ОБЪЕМА ===
DEFAULT_REAL_WIDTH = 10.0  # mm
DEFAULT_REAL_HEIGHT = 2.0  # mm
SCAN_NUMBER_MIN = 1
SCAN_NUMBER_MAX = 99  # Обновлено для поддержки двузначных чисел
DELAUNAY_ALPHA = 50.0
CONTOUR_APPROX_RATE = 0.002
CONTOUR_MIN_POINTS = 4
VOLUME_DIVIDER = 1000.0
TARGET_MIN_ANGLE_STEP = 4.5 # degrees

# Настройка логирования
# Для совместимости с Python < 3.9, открываем файл явно с utf-8 кодировкой
log_file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
logging.basicConfig(handlers=[log_file_handler], level=LOG_LEVEL, format=LOG_FORMAT) # Removed encoding argument, added handlers

def show_error(message: str, level: str = 'critical'):
    """Универсальная функция для отображения ошибок и логирования."""
    level_actions = {
        'critical': (logging.error, QMessageBox.critical, "CRITICAL ERROR"),
        'warning': (logging.warning, QMessageBox.warning, "WARNING")
    }

    log_func, dialog_func, console_prefix = level_actions[level]
    log_func(message)

    app = QApplication.instance()
    if app is not None:
        dialog_func(None, "Ошибка" if level == 'critical' else "Внимание", message)
    else:
        print(f"{console_prefix}: {message}")

class DataReader:
    def __init__(self, directory, templates_dir=TEMPLATES_DIR):
        self.directory = Path(directory)
        self.templates_dir = Path(templates_dir)
        self.images = []
        self.scan_numbers = []
        self.digit_templates = {}  # Шаблоны для отдельных цифр 0-9
        self.image_shape = None  # (height, width, channels)
        self._load_digit_templates_from_templates()

    def _imread_unicode(self, path):
        try:
            with open(path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Не удалось декодировать изображение: {path}")
            return img
        except Exception as e:
            logging.error(f"Ошибка чтения изображения {path}: {str(e)}")
            return None

    def _load_digit_templates_from_templates(self):
        """Загружает шаблоны для цифр 0-9 из файлов."""
        self.digit_templates = {}

        # Загружаем шаблоны для всех цифр 0-9
        for digit in range(10):
            template_path = self.templates_dir / f"{digit}.png"
            template_bgr = self._imread_unicode(template_path)
            if template_bgr is None:
                show_error(f"Шаблон для цифры {digit} не найден: {template_path}", level='warning')
                continue
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            bbox = self._find_number_bbox(template_gray)
            normalized_template = self._extract_and_normalize_number(template_gray, bbox)
            if normalized_template is not None:
                self.digit_templates[digit] = normalized_template

        if not self.digit_templates:
            show_error("Не удалось загрузить ни один шаблон цифры")
            self.digit_templates = None
        else:
            logging.info(f"Загружено {len(self.digit_templates)} шаблонов цифр: {sorted(self.digit_templates.keys())}")

    def _find_number_bbox(self, gray_roi):
        """Находит ограничивающий прямоугольник для числа."""
        try:
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                show_error("Контуры числа не найдены", level='warning')
                return None

            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
            if not valid_contours:
                show_error("Валидные контуры числа не найдены", level='warning')
                return None

            # Находим общие границы всех контуров
            all_points = np.vstack(valid_contours).squeeze()
            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)

            return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        except Exception as e:
            show_error(f"Ошибка поиска контура числа: {str(e)}")
            return None

    def _extract_and_normalize_number(self, gray_roi, bbox):
        """Извлекает и нормализует изображение числа."""
        try:
            if bbox is None:
                return None
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                show_error(f"Некорректные размеры bbox: w={w}, h={h}")
                return None
            number_img = gray_roi[y:y + h, x:x + w]
            if number_img.size == 0:
                show_error(f"Пустое изображение числа после вырезки bbox")
                return None
            normalized_number = cv2.resize(number_img, TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
            return normalized_number
        except Exception as e:
            show_error(f"Ошибка нормализации числа: {str(e)}")
            return None

    def _find_number_roi(self, img_bgr):
        """Вырезает ROI с номером скана из левого верхнего угла."""
        h, w = img_bgr.shape[:2]
        roi_h = int(h * 0.2)
        roi_w = int(w * 0.2)
        roi_bgr = img_bgr[0:roi_h, 0:roi_w]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return roi_bgr, roi_gray

    def _extract_digits_from_roi(self, roi_gray):
        """Извлекает одну или две цифры из ROI номера."""
        # Бинаризация для выделения белых цифр
        _, thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # bounding boxes по x
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        if not bboxes:
            return []
        bboxes = sorted(bboxes, key=lambda b: b[0])
        digit_imgs = []
        for x, y, w, h in bboxes:
            digit = roi_gray[y:y+h, x:x+w]
            norm_digit = cv2.resize(digit, TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
            digit_imgs.append(norm_digit)
        return digit_imgs

    def _recognize_digit(self, digit_img):
        if digit_img is None or self.digit_templates is None or not self.digit_templates:
            show_error("Нет изображения цифры или шаблонов", level='warning')
            return None, None
        best_match_digit = None
        best_match_value = -1.0
        for digit, template in self.digit_templates.items():
            if digit_img.shape != template.shape:
                continue
            res = cv2.matchTemplate(digit_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_match_value:
                best_match_value = max_val
                best_match_digit = digit
        if best_match_value < CONFIDENCE_THRESHOLD:
            show_error(f"Цифра не распознана, уверенность: {best_match_value:.2f}", level='warning')
            return None, best_match_value
        return best_match_digit, best_match_value

    def _extract_number(self, roi_gray):
        try:
            digit_imgs = self._extract_digits_from_roi(roi_gray)
            if not digit_imgs:
                return None
            digits = []
            for digit_img in digit_imgs:
                digit, conf = self._recognize_digit(digit_img)
                if digit is not None:
                    digits.append(digit)
            if not digits:
                return None
            # Если одна цифра — однозначное число, если две — двузначное
            if len(digits) == 1:
                return digits[0]
            elif len(digits) == 2:
                return digits[0] * 10 + digits[1]
            else:
                # Если больше двух, берём две самых левых
                return digits[0] * 10 + digits[1]
        except Exception as e:
            show_error(f"Ошибка извлечения номера: {str(e)}")
            return None

    def read_images(self):
        try:
            self.images = []
            self.scan_numbers = []
            self.image_files = [] # Store file paths here
            seen_numbers = set()
            image_shape = None
            all_image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                all_image_paths.extend(self.directory.glob(ext))

            # Sort files by name to ensure consistent order (e.g., scan_1.png, scan_2.png)
            all_image_paths = sorted(all_image_paths, key=lambda p: p.name)

            for file_path in all_image_paths:
                img_bgr = self._imread_unicode(file_path)
                if img_bgr is None:
                    continue
                roi_bgr, roi_gray = self._find_number_roi(img_bgr)
                number = self._extract_number(roi_gray)
                if number is None or not (SCAN_NUMBER_MIN <= number <= SCAN_NUMBER_MAX):
                    show_error(f"Неверный номер скана в файле: {file_path.name}", level='warning')
                    continue
                if number in seen_numbers:
                    show_error(f"Обнаружен дубликат номера скана {number} в файле: {file_path.name}", level='warning')
                    continue
                if img_bgr is None:
                    continue
                if image_shape is None:
                    image_shape = img_bgr.shape
                elif img_bgr.shape != image_shape:
                    show_error(f"Обнаружено изображение с другим разрешением: {file_path.name} (ожидалось {image_shape}, получено {img_bgr.shape})")
                    raise ValueError(f"Обнаружено изображение с другим разрешением: {file_path.name} (ожидалось {image_shape}, получено {img_bgr.shape})")
                self.images.append(img_bgr)
                self.scan_numbers.append(number)
                self.image_files.append(file_path) # Store the path
                seen_numbers.add(number)

            # After collecting all valid images, sort them by scan number
            # Create a list of tuples (scan_number, image, file_path) and sort it
            sorted_data = sorted(zip(self.scan_numbers, self.images, self.image_files))
            self.scan_numbers = [item[0] for item in sorted_data]
            self.images = [item[1] for item in sorted_data]
            self.image_files = [item[2] for item in sorted_data]

            missing = set(range(min(self.scan_numbers), max(self.scan_numbers) + 1)) - set(self.scan_numbers) if self.scan_numbers else set()
            if missing:
                show_error(f"Отсутствуют сканы: {sorted(list(missing))}", level='warning')
            if not self.images:
                show_error("Не найдено ни одного валидного изображения")
                raise ValueError("Не найдено ни одного валидного изображения")
            self.image_shape = image_shape
            return self.images, self.scan_numbers, image_shape
        except Exception as e:
            show_error(f"Ошибка чтения данных: {str(e)}")
            raise


class ImageProcessor:
    def __init__(self, saturation_threshold=0.1225):
        self.saturation_threshold = saturation_threshold
    @staticmethod
    def process_image(img, approximation_rate=CONTOUR_APPROX_RATE, saturation_threshold=0.0):
        """Выделяет контур из изображения."""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > (saturation_threshold * 255)
            h, w = mask.shape
            kernel_size = min(MORPH_KERNEL_MAX_SIZE, h, w)
            if kernel_size < 1:
                show_error("Размер ядра морфологии слишком мал")
                return None
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if len(contour) < CONTOUR_MIN_POINTS:
                    show_error("Контур слишком мал", level='warning')
                    return None
                arclen = cv2.arcLength(contour, True)
                epsilon = arclen * approximation_rate
                approx = cv2.approxPolyDP(contour, epsilon, True)
                return approx
            else:
                show_error("Контур не найден", level='warning')
                return None
        except Exception as e:
            show_error(f"Ошибка обработки изображения: {str(e)}")
            return None


class ModelBuilder:
    def __init__(self, image_width, image_height, real_width=DEFAULT_REAL_WIDTH, real_height=DEFAULT_REAL_HEIGHT, n_resample_points=100):
        self.points = None  # numpy array (N, 3)
        self.mesh = None    # pyvista.PolyData
        self.volume = 0.0
        self.IMAGE_WIDTH = image_width  # pixels
        self.IMAGE_HEIGHT = image_height  # pixels
        self.REAL_WIDTH = real_width   # mm
        self.REAL_HEIGHT = real_height   # mm
        self.scale_x = self._calculate_default_scale(self.IMAGE_WIDTH, self.REAL_WIDTH)
        self.scale_y = self._calculate_default_scale(self.IMAGE_HEIGHT, self.REAL_HEIGHT)
        self.n_resample_points = n_resample_points
        logging.info(f"Default scales calculated: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm")

    def _calculate_default_scale(self, image_dim, real_dim):
        """Вычисляет масштаб по умолчанию."""
        return image_dim / real_dim if real_dim != 0 else 1.0

    def set_scale(self, scale_x, scale_y):
        self.scale_x = scale_x if scale_x > 0 else self._calculate_default_scale(self.IMAGE_WIDTH, self.REAL_WIDTH)
        self.scale_y = scale_y if scale_y > 0 else self._calculate_default_scale(self.IMAGE_HEIGHT, self.REAL_HEIGHT)
        logging.info(f"Scale set: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm")

    def _perform_interpolation_step(self, existing_contours, existing_angles):
        """
        Performs one step of interpolation, effectively halving the angular resolution
        by creating new contours in between existing ones.
        """
        try:
            if not existing_contours or not existing_angles:
                logging.warning("Cannot interpolate: no contours or angles provided.")
                return existing_contours, existing_angles

            if len(existing_contours) < 2:
                logging.info("Only one contour, no interpolation needed.")
                return existing_contours, existing_angles

            # Determine the current average angular step between existing contours
            # Assuming existing_angles are sorted and represent a full circle.
            current_avg_angle_step = 360.0 / len(existing_angles)

            # Calculate the new angular step (halved)
            new_angular_step = current_avg_angle_step / 2.0

            # Determine the total number of new angles for a full circle
            total_new_angles_count = int(round(360.0 / new_angular_step))

            # Generate the list of all target angles for the full circle
            # Use linspace to ensure even distribution and avoid floating point issues near 360
            new_full_angles = np.linspace(0, 360, total_new_angles_count, endpoint=False) # Exclude 360 to avoid duplicate 0/360

            # Create a new list for contours at the new angles, initialized with None
            new_all_contours = [None] * total_new_angles_count

            # Map existing contours to their correct positions in the new, denser angle list
            for i, existing_contour in enumerate(existing_contours):
                existing_angle = existing_angles[i]
                # Calculate the expected index in new_full_angles
                expected_index = int(round(existing_angle / new_angular_step))
                expected_index = expected_index % total_new_angles_count # Ensure it wraps around for 360/0

                # Check for collision (should not happen if initial angles are perfectly spaced)
                if new_all_contours[expected_index] is not None:
                    logging.warning(f"Collision detected at index {expected_index} ({new_full_angles[expected_index]:.2f} deg). Overwriting existing contour from {existing_angle:.2f}.")
                new_all_contours[expected_index] = existing_contour

            # Perform interpolation for missing contours (where new_all_contours is None)
            final_interpolated_contours = []
            final_interpolated_angles = []

            for i in range(total_new_angles_count):
                if new_all_contours[i] is None:
                    # Find nearest valid previous and next contours circularly
                    prev_idx = None
                    for j in range(1, total_new_angles_count + 1):
                        k = (i - j + total_new_angles_count) % total_new_angles_count
                        if new_all_contours[k] is not None:
                            prev_idx = k
                            break

                    next_idx = None
                    for j in range(1, total_new_angles_count + 1):
                        k = (i + j) % total_new_angles_count
                        if new_all_contours[k] is not None:
                            next_idx = k
                            break

                    if prev_idx is not None and next_idx is not None and prev_idx != next_idx:
                        c1 = new_all_contours[prev_idx]
                        c2 = new_all_contours[next_idx]

                        angle1 = new_full_angles[prev_idx]
                        angle2 = new_full_angles[next_idx]
                        target_angle = new_full_angles[i]

                        # Handle circular wrap-around for angles (e.g., 350 to 0 degrees)
                        # Adjust angle2 and target_angle if crossing 0/360 for correct interpolation 't' value
                        if angle2 < angle1:
                            angle2 += 360.0
                            if target_angle < angle1:
                                target_angle += 360.0

                        if abs(angle2 - angle1) < 1e-9: # Prevent division by zero if angles are too close
                            logging.warning(f"Interpolation skipped: angles too close for index {i}. Using previous contour as fallback.")
                            interp_contour = c1 # Fallback to a valid contour
                        else:
                            t = (target_angle - angle1) / (angle2 - angle1)
                            interp_contour = self._interpolate_contour(c1, c2, t)

                        final_interpolated_contours.append(interp_contour)
                        final_interpolated_angles.append(new_full_angles[i])
                    else:
                        logging.warning(f"Cannot find sufficient neighbors to interpolate contour for angle {new_full_angles[i]:.2f}. This angle will be skipped.")
                        # If no interpolation possible, we just skip this angle.
                else:
                    final_interpolated_contours.append(new_all_contours[i])
                    final_interpolated_angles.append(new_full_angles[i])

            if len(final_interpolated_contours) == 0:
                show_error("After interpolation, no valid contours remain. This indicates a severe issue.", level='critical')
                raise ValueError("No valid contours after interpolation.")

            logging.info(f"Interpolation step completed. From {len(existing_contours)} to {len(final_interpolated_contours)} contours with new angular step {new_angular_step:.2f} degrees.")
            return final_interpolated_contours, final_interpolated_angles

        except Exception as e:
            logging.error(f"Ошибка в _perform_interpolation_step: {str(e)}", exc_info=True)
            show_error(f"Ошибка интерполяции контуров: {str(e)}")
            return existing_contours, existing_angles

    def _resample_contour(self, contour, n_points=None):
        if n_points is None:
            n_points = self.n_resample_points if hasattr(self, 'n_resample_points') else 100
        # Преобразует контур в массив shape (N, 2)
        pts = contour.squeeze()
        if len(pts.shape) == 1:
            pts = pts[None, :]
        if pts.shape[0] < 2:
            return contour
        # Вычисляем длины сегментов
        dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        dists = np.insert(dists, 0, 0)
        cumulative = np.cumsum(dists)
        total_length = cumulative[-1]
        if total_length == 0:
            return contour
        # Новые равномерные позиции вдоль длины
        even_spaced = np.linspace(0, total_length, n_points)
        new_pts = []
        for t in even_spaced:
            idx = np.searchsorted(cumulative, t)
            if idx == 0:
                new_pts.append(pts[0])
            elif idx >= len(pts):
                new_pts.append(pts[-1])
            else:
                t0, t1 = cumulative[idx-1], cumulative[idx]
                p0, p1 = pts[idx-1], pts[idx]
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
                new_pt = (1 - alpha) * p0 + alpha * p1
                new_pts.append(new_pt)
        new_pts = np.array(new_pts, dtype=np.int32)
        return new_pts.reshape(-1, 1, 2)

    def _interpolate_contour(self, c1, c2, t):
        try:
            c1r = self._resample_contour(c1)
            c2r = self._resample_contour(c2)
            interp_points = (c1r.astype(np.float32) * (1 - t) + c2r.astype(np.float32) * t).astype(np.int32)
            return interp_points
        except Exception as e:
            logging.error(f"Ошибка интерполяции контура: {str(e)}")
            return c1

    def build_model(self, contours, scan_numbers, angles=None, center=None, delaunay_alpha=None):
        try:
            if not contours or not scan_numbers:
                show_error("Нет валидных контуров или номеров сканов")
                raise ValueError("Нет валидных сканов для вычисления углов")

            # Вычисляем углы если не заданы
            if angles is None:
                N = len(scan_numbers)
                if N == 0:
                    show_error("Нет валидных сканов для вычисления углов")
                    raise ValueError("Нет валидных сканов для вычисления углов")
                if 360 % N != 0:
                    show_error(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом.")
                    raise ValueError(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом.")
                initial_angle_step = 360.0 / N
                initial_angles = [i * initial_angle_step for i in range(N)]
            else:
                initial_angles = angles[:]

            # Интерполяция контуров
            current_contours = contours[:]
            current_angles = initial_angles[:]

            if len(current_angles) > 1:
                current_effective_angle_step = sorted(current_angles)[1] - sorted(current_angles)[0]
            else:
                current_effective_angle_step = 360.0

            # Цикл интерполяции
            interpolation_iteration = 0
            while current_effective_angle_step > TARGET_MIN_ANGLE_STEP + 1e-9:
                interpolation_iteration += 1
                logging.info(f"Interpolation iteration {interpolation_iteration}: Current effective angle step {current_effective_angle_step:.2f} degrees. Target: {TARGET_MIN_ANGLE_STEP:.2f} degrees.")

                new_contours, new_angles = self._perform_interpolation_step(current_contours, current_angles)

                if len(new_contours) == len(current_contours):
                    logging.info("Interpolation step did not increase contour density. Stopping interpolation.")
                    break

                current_contours = new_contours
                current_angles = new_angles

                if len(current_angles) > 1:
                    current_effective_angle_step = sorted(current_angles)[1] - sorted(current_angles)[0]
                else:
                    logging.warning("Only one contour remains after interpolation. Stopping.")
                    break

                if interpolation_iteration > 15:
                    logging.warning("Too many interpolation iterations, potential infinite loop detected. Stopping.")
                    break

            contours = current_contours
            angles = current_angles
            scan_numbers = list(range(1, len(contours) + 1))
            logging.info(f"Final number of contours after interpolation: {len(contours)}, with angular step: {current_effective_angle_step:.2f} degrees.")

            # Создание 3D точек
            if center is None:
                center = (self.IMAGE_WIDTH // 2, self.IMAGE_HEIGHT // 2)

            points_list = []
            for i, contour in enumerate(contours):
                angle_rad = angles[i] * np.pi / 180
                for point in contour:
                    x, y = point[0]
                    x_physical = (x - center[0]) / self.scale_x
                    y_physical = (center[1] - y) / self.scale_y

                    x_3d = x_physical * np.cos(angle_rad)
                    y_3d = y_physical
                    z_3d = x_physical * np.sin(angle_rad)

                    points_list.append([x_3d, y_3d, z_3d])

            points = np.array(points_list)
            unique_points = np.unique(points, axis=0)
            logging.info(f"Всего точек: {points.shape[0]}, уникальных: {unique_points.shape[0]}")
            if points.shape[0] < 4:
                show_error(f"Недостаточно точек для триангуляции: {points.shape[0]}")
                raise ValueError(f"Недостаточно точек для триангуляции: {points.shape[0]}")

            self.points = points

            # Создание 3D модели
            current_delaunay_alpha = delaunay_alpha if delaunay_alpha is not None else DELAUNAY_ALPHA
            cloud = pv.PolyData(points)
            mesh = cloud.delaunay_3d(alpha=current_delaunay_alpha)
            surf = mesh.extract_geometry()
            logging.info(f"Full mesh: {surf.n_faces} faces, volume: {surf.volume:.2f}")
            self.mesh = surf
            self.volume = surf.volume
            return surf

        except Exception as e:
            show_error(f"Ошибка построения модели: {str(e)}")
            raise


class DebugViewer(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Отладочный просмотрщик изображений")
        self.setGeometry(200, 200, 1000, 700)
        
        # Данные для отображения
        self.images = []
        self.scan_numbers = []
        self.contours = []
        self.image_files = []
        self.current_index = 0
        
        # Настройка UI
        self.init_ui()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Информационная панель
        info_layout = QtWidgets.QHBoxLayout()
        
        self.info_label = QtWidgets.QLabel("Нет данных")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        info_layout.addWidget(self.info_label)
        
        # Кнопки навигации
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.prev_button = QtWidgets.QPushButton("← Предыдущее")
        self.prev_button.clicked.connect(self.show_previous)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QtWidgets.QPushButton("Следующее →")
        self.next_button.clicked.connect(self.show_next)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        info_layout.addLayout(nav_layout)
        layout.addLayout(info_layout)
        
        # Графическая сцена для отображения изображения
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        layout.addWidget(self.view)
        
        # Панель с дополнительной информацией
        details_layout = QtWidgets.QHBoxLayout()
        
        self.details_label = QtWidgets.QLabel("Детали распознавания:")
        self.details_label.setStyleSheet("font-size: 12px;")
        details_layout.addWidget(self.details_label)
        
        layout.addLayout(details_layout)
        
        # Подсказки для пользователя
        help_layout = QtWidgets.QHBoxLayout()
        
        help_text = ("Подсказки: ←/→ навигация, колесико мыши - масштаб, 0 - сброс масштаба, "
                    "Esc - закрыть, перетаскивание мышью - перемещение")
        help_label = QtWidgets.QLabel(help_text)
        help_label.setStyleSheet("font-size: 10px; color: gray; font-style: italic;")
        help_layout.addWidget(help_label)
        
        layout.addLayout(help_layout)
        
    def set_data(self, images, scan_numbers, contours, image_files):
        """Устанавливает данные для отображения."""
        self.images = images
        self.scan_numbers = scan_numbers
        self.contours = contours
        self.image_files = image_files
        self.current_index = 0
        
        if self.images:
            self.update_navigation_buttons()
            self.show_current_image()
        else:
            self.info_label.setText("Нет данных для отображения")
            
    def update_navigation_buttons(self):
        """Обновляет состояние кнопок навигации."""
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.images) - 1)
        
    def show_current_image(self):
        """Отображает текущее изображение с контуром и информацией."""
        if not self.images or self.current_index >= len(self.images):
            return
            
        # Получаем данные текущего изображения
        img = self.images[self.current_index]
        scan_number = self.scan_numbers[self.current_index] if self.current_index < len(self.scan_numbers) else "N/A"
        contour = self.contours[self.current_index] if self.current_index < len(self.contours) else None
        file_name = self.image_files[self.current_index] if self.current_index < len(self.image_files) else "unknown"
        
        # Создаем копию изображения для рисования
        display_img = img.copy()
        
        # Рисуем ROI (область интереса) - левый верхний угол
        h, w = img.shape[:2]
        roi_h = int(h * 0.2)
        roi_w = int(w * 0.2)
        cv2.rectangle(display_img, (0, 0), (roi_w, roi_h), (0, 255, 255), 2)
        cv2.putText(display_img, "ROI", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Рисуем контур если есть
        if contour is not None:
            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
            
        # Конвертируем в RGB для Qt
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Создаем QPixmap
        h, w, ch = display_img_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(display_img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        
        # Очищаем сцену и добавляем новое изображение
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # Обновляем информацию
        self.info_label.setText(f"Изображение {self.current_index + 1}/{len(self.images)} | "
                              f"Номер скана: {scan_number} | "
                              f"Файл: {Path(file_name).name}")
        
        # Детали распознавания
        details = []
        if contour is not None:
            details.append(f"Контур найден: {len(contour)} точек")
            area = cv2.contourArea(contour)
            details.append(f"Площадь контура: {area:.1f} пикселей²")
            
            # Вычисляем периметр контура
            perimeter = cv2.arcLength(contour, True)
            details.append(f"Периметр: {perimeter:.1f} пикселей")
            
            # Вычисляем компактность (отношение площади к квадрату периметра)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                details.append(f"Компактность: {compactness:.3f}")
        else:
            details.append("Контур не найден")
            
        if scan_number != "N/A":
            details.append(f"Распознан номер: {scan_number}")
        else:
            details.append("Номер не распознан")
            
        # Добавляем информацию о размере изображения
        details.append(f"Размер: {w}x{h} пикселей")
            
        self.details_label.setText(" | ".join(details))
        
    def show_previous(self):
        """Показывает предыдущее изображение."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_navigation_buttons()
            self.show_current_image()
            
    def show_next(self):
        """Показывает следующее изображение."""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.update_navigation_buttons()
            self.show_current_image()
            
    def resizeEvent(self, event):
        """Обработчик изменения размера окна."""
        super().resizeEvent(event)
        if self.scene and not self.scene.sceneRect().isEmpty():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            
    def keyPressEvent(self, event):
        """Обработчик нажатий клавиш для навигации."""
        if event.key() == Qt.Key.Key_Left:
            self.show_previous()
        elif event.key() == Qt.Key.Key_Right:
            self.show_next()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_0:  # Сброс масштаба
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            super().keyPressEvent(event)
            
    def wheelEvent(self, event):
        """Обработчик колесика мыши для масштабирования."""
        if event.angleDelta().y() > 0:
            self.view.scale(1.1, 1.1)
        else:
            self.view.scale(0.9, 0.9)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Scan Processor (PyVista)")
        self.setGeometry(100, 100, 800, 600)
        self.reader = DataReader('.')
        self.processor = ImageProcessor()
        self.builder = None
        self.plotter = None
        self.progress_bar = None
        self.resample_points = 100  # Новый параметр по умолчанию
        self.delaunay_alpha = DELAUNAY_ALPHA  # Новый параметр по умолчанию
        self.debug_viewer = DebugViewer(self)
        self.init_ui()

    def init_ui(self):
        # Создаем меню-бар
        self.create_menu_bar()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.vtk_widget = QtInteractor(central_widget)
        layout.addWidget(self.vtk_widget.interactor)

        self.volume_label = QtWidgets.QLabel("Объём: N/A")
        self.volume_label.setStyleSheet("font-size: 16px; color: blue;")
        self.volume_label.mousePressEvent = self.copy_volume
        layout.addWidget(self.volume_label)

        # Добавляем прогресс-бар
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def create_menu_bar(self):
        """Создает меню-бар с действиями."""
        menubar = self.menuBar()

        # Меню "Файл"
        file_menu = menubar.addMenu('&Файл')

        # Действие "Открыть папку"
        open_action = QAction('&Открыть папку...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Выбрать папку с изображениями сканов')
        open_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_action)

        # Новый пункт: Открыть одиночное изображение
        open_single_action = QAction('&Открыть изображение...', self)
        open_single_action.setShortcut('Ctrl+I')
        open_single_action.setStatusTip('Открыть отдельное изображение и показать распознанный контур')
        open_single_action.triggered.connect(self.open_single_image)
        file_menu.addAction(open_single_action)

        # Разделитель
        file_menu.addSeparator()

        # Действие "Отладочный просмотрщик"
        debug_action = QAction('&Отладочный просмотрщик...', self)
        debug_action.setShortcut('Ctrl+D')
        debug_action.setStatusTip('Открыть отладочный просмотрщик для проверки распознавания')
        debug_action.triggered.connect(self.open_debug_viewer)
        file_menu.addAction(debug_action)

        # Разделитель
        file_menu.addSeparator()

        # Действие "Выход"
        exit_action = QAction('&Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Выйти из приложения')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню "Инструменты"
        tools_menu = menubar.addMenu('&Инструменты')

        # Действие "Настройки масштаба"
        settings_action = QAction('&Настройки масштаба...', self)
        settings_action.setShortcut('Ctrl+S')
        settings_action.setStatusTip('Настроить масштаб модели')
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)

        # Меню "Справка"
        help_menu = menubar.addMenu('&Справка')

        # Действие "О программе"
        about_action = QAction('&О программе', self)
        about_action.setStatusTip('Информация о программе')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        """Показывает диалог "О программе"."""
        QtWidgets.QMessageBox.about(
            self,
            "О программе",
            "3D Scan Processor\n\n"
            "Программа для обработки 3D сканов и расчета объема\n"
            "Использует PyVista для 3D визуализации\n\n"
            "Версия 1.0"
        )

    def _set_progress(self, visible: bool, maximum: int = 100, value: int = 0, text: str = ""):
        if self.progress_bar is not None:
            self.progress_bar.setVisible(visible)
            if text:
                self.progress_bar.setFormat(text)
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
            QtWidgets.QApplication.processEvents()

    def _find_optimal_alpha(self, contours, scan_numbers, angles, initial_delaunay_alpha):
        logging.info("Starting optimal DELAUNAY_ALPHA search using adaptive approach...")
        try:
            low_alpha_bound = 10.0
            high_alpha_bound = 250.0
            best_manifold_alpha_in_range = None
            binary_search_iterations = 3
            linear_scan_steps = 3
            total_progress_steps = binary_search_iterations + linear_scan_steps
            self._set_progress(True, total_progress_steps, 0, "Поиск Alpha (фаза 1/2): %p%")
            for i in range(binary_search_iterations):
                current_alpha = (low_alpha_bound + high_alpha_bound) / 2
                try:
                    if self.builder is None:
                        logging.error("ModelBuilder is not initialized in _find_optimal_alpha.")
                        raise ValueError("ModelBuilder not initialized.")
                    model = self.builder.build_model(contours, scan_numbers, angles=angles, delaunay_alpha=current_alpha)
                    is_manifold = model.is_manifold
                    volume = model.volume
                    logging.info(f"  Binary Alpha={current_alpha:.2f}: Manifold={is_manifold}, Volume={volume:.2f}")
                    if is_manifold and volume > 0:
                        best_manifold_alpha_in_range = current_alpha
                        high_alpha_bound = current_alpha
                    else:
                        low_alpha_bound = current_alpha
                except ValueError as e:
                    logging.warning(f"  Binary Search: Failed to build model for Alpha={current_alpha:.2f}: {e}")
                    low_alpha_bound = current_alpha
                except Exception as e:
                    logging.error(f"  Binary Search: Unexpected error for Alpha={current_alpha:.2f}: {e}", exc_info=True)
                    low_alpha_bound = current_alpha
                self._set_progress(True, total_progress_steps, i + 1)
            if best_manifold_alpha_in_range is None:
                logging.warning(f"Binary search failed to find a manifold alpha in range [1.0, 1000.0]. Reverting to a broader linear scan starting from initial_delaunay_alpha.")
                search_start = max(1.0, initial_delaunay_alpha - 100)
                search_end = initial_delaunay_alpha + 300
                alpha_values_for_linear_scan = np.linspace(search_start, search_end, linear_scan_steps)
                best_manifold_alpha_in_range = initial_delaunay_alpha
            else:
                logging.info(f"Phase 1 complete. Smallest manifold alpha found (approx): {best_manifold_alpha_in_range:.2f}")
                search_start = max(1.0, best_manifold_alpha_in_range - 50)
                search_end = best_manifold_alpha_in_range + 200
                if search_end > 1000.0:
                    search_end = 1000.0
                if search_start >= search_end:
                    search_start = max(1.0, search_end - 100)
                alpha_values_for_linear_scan = np.linspace(search_start, search_end, linear_scan_steps)
                logging.info(f"Phase 2: Linear scan in refined range: [{search_start:.2f}, {search_end:.2f}] with {linear_scan_steps} steps.")
            candidate_alpha_results = []
            self._set_progress(True, total_progress_steps, binary_search_iterations, "Поиск Alpha (фаза 2/2): %p%")
            for idx, current_alpha in enumerate(alpha_values_for_linear_scan):
                try:
                    model = self.builder.build_model(contours, scan_numbers, angles=angles, delaunay_alpha=current_alpha)
                    volume = model.volume
                    n_faces = model.n_faces
                    is_manifold = model.is_manifold
                    logging.info(f"  Linear Alpha={current_alpha:.2f}: Volume={volume:.2f} мм³, Faces={n_faces}, Manifold={is_manifold}")
                    if is_manifold and volume > 0:
                        candidate_alpha_results.append({'alpha': current_alpha, 'volume': volume})
                except ValueError as e:
                    logging.warning(f"  Linear Scan: Failed to build model for Alpha={current_alpha:.2f}: {e}")
                except Exception as e:
                    logging.error(f"  Linear Scan: Unexpected error for Alpha={current_alpha:.2f}: {e}", exc_info=True)
                self._set_progress(True, total_progress_steps, binary_search_iterations + idx + 1)
        finally:
            self._set_progress(False)
        if not candidate_alpha_results:
            logging.warning("No suitable alpha value found to create a manifold mesh in refined scan. Using default DELAUNAY_ALPHA.")
            return initial_delaunay_alpha
        volumes = [r['volume'] for r in candidate_alpha_results]
        if not volumes:
            logging.warning("No valid volumes found among manifold candidates. Using default DELAUNAY_ALPHA.")
            return initial_delaunay_alpha
        median_volume = np.median(volumes)
        best_alpha_candidate = None
        min_diff_from_median = float('inf')
        for r in candidate_alpha_results:
            current_diff = abs(r['volume'] - median_volume)
            if current_diff < min_diff_from_median:
                min_diff_from_median = current_diff
                best_alpha_candidate = r['alpha']
        logging.info(f"Optimal DELAUNAY_ALPHA selected: {best_alpha_candidate:.2f} (median volume of candidates: {median_volume:.2f} мм³)")
        return best_alpha_candidate

    def select_folder(self):
        try:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
            if folder:
                self.reader.directory = Path(folder)
                images, scan_numbers, image_shape = self.reader.read_images()
                if image_shape is None:
                    show_error("Не удалось определить разрешение изображений")
                    raise ValueError("Не удалось определить разрешение изображений")
                image_height, image_width = image_shape[:2]
                N = len(scan_numbers)
                if N == 0:
                    show_error("Не найдено ни одного валидного изображения для построения модели")
                    raise ValueError("Не найдено ни одного валидного изображения для построения модели")
                if 360 % N != 0:
                    show_error(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом. Попробуйте другое количество кадров.")
                    raise ValueError(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом. Попробуйте другое количество кадров.")
                angle = 360 // N
                angles = [i * angle for i in range(N)]
                self.builder = ModelBuilder(image_width, image_height)
                self._set_progress(True, len(images), 0, "Обработка изображений: %p%")
                contours = []
                for idx, img in enumerate(images):
                    contour = ImageProcessor.process_image(img, saturation_threshold=self.processor.saturation_threshold)
                    if contour is not None:
                        contours.append(contour)
                    self._set_progress(True, len(images), idx + 1)
                self._set_progress(False)
                if not contours:
                    show_error("Не удалось извлечь ни одного контура")
                    raise ValueError("Не удалось извлечь ни одного контура")
                self.last_contours = contours
                self.last_scan_numbers = scan_numbers
                self.last_angles = angles
                self.last_images = images  # Сохраняем изображения для отладки
                self.last_image_files = [f.name for f in self.reader.image_files] # Store just the names
                optimal_alpha = self._find_optimal_alpha(contours, scan_numbers, angles, DELAUNAY_ALPHA)
                show_error(f"Optimal DELAUNAY_ALPHA: {optimal_alpha:.2f}")
                model = self.builder.build_model(contours, scan_numbers, angles=angles, delaunay_alpha=optimal_alpha)
                self.visualize_model(model)
                volume_mm3 = self.builder.volume
                volume_ml = volume_mm3 / VOLUME_DIVIDER
                self.volume_label.setText(f"Объём: {volume_mm3:.4f} мм³ ({volume_ml:.5f} мл)")
        except Exception as e:
            show_error(f"Ошибка обработки: {str(e)}")
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)

    def open_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки масштаба")
        layout = QtWidgets.QVBoxLayout(dialog)

        if self.builder is None:
            show_error("Модель не инициализирована. Сначала выберите папку с изображениями.")
            return

        # Add information about image and real dimensions
        info_label = QtWidgets.QLabel(
            f"Размеры изображения: {self.builder.IMAGE_WIDTH}x{self.builder.IMAGE_HEIGHT} пикселей\n"
            f"Реальные размеры: {self.builder.REAL_WIDTH}x{self.builder.REAL_HEIGHT} мм"
        )
        layout.addWidget(info_label)

        # Add current scale information
        current_scale_label = QtWidgets.QLabel(
            f"Текущий масштаб:\n"
            f"X,Z: {self.builder.scale_x:.2f} пикселей/мм\n"
            f"Y: {self.builder.scale_y:.2f} пикселей/мм"
        )
        layout.addWidget(current_scale_label)

        # Add scale input fields
        layout.addWidget(QtWidgets.QLabel("Масштаб по X и Z (пикселей на мм):"))
        scale_x_input = QtWidgets.QLineEdit()
        scale_x_input.setText(f"{self.builder.scale_x:.2f}")
        layout.addWidget(scale_x_input)

        layout.addWidget(QtWidgets.QLabel("Масштаб по Y (пикселей на мм):"))
        scale_y_input = QtWidgets.QLineEdit()
        scale_y_input.setText(f"{self.builder.scale_y:.2f}")
        layout.addWidget(scale_y_input)

        # Новое поле для количества точек ремэппинга
        layout.addWidget(QtWidgets.QLabel("Количество точек на контуре (ремэппинг):"))
        resample_points_input = QtWidgets.QLineEdit()
        resample_points_input.setText(str(self.resample_points))
        layout.addWidget(resample_points_input)

        # Новое поле для DELAUNAY_ALPHA
        layout.addWidget(QtWidgets.QLabel("DELAUNAY ALPHA (параметр триангуляции):"))
        delaunay_alpha_input = QtWidgets.QLineEdit()
        delaunay_alpha_input.setText(str(self.delaunay_alpha))
        layout.addWidget(delaunay_alpha_input)

        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()

        reset_button = QtWidgets.QPushButton("Сбросить")
        reset_button.clicked.connect(lambda: self._reset_scales(scale_x_input, scale_y_input))
        button_layout.addWidget(reset_button)

        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)

        # Выполняем диалог и обрабатываем результат только если нажали OK
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                scale_x = float(scale_x_input.text())
                scale_y = float(scale_y_input.text())
                resample_points = int(resample_points_input.text())
                delaunay_alpha = float(delaunay_alpha_input.text())
                if resample_points < 4:
                    resample_points = 4
                if delaunay_alpha < 1.0:
                    delaunay_alpha = 1.0
                self.resample_points = resample_points
                self.delaunay_alpha = delaunay_alpha
                self.builder.set_scale(scale_x, scale_y)
                # Перестраиваем модель с новыми масштабами, количеством точек и alpha
                if hasattr(self, 'last_contours') and hasattr(self, 'last_scan_numbers') and hasattr(self, 'last_angles'):
                    self.builder.n_resample_points = self.resample_points
                    model = self.builder.build_model(self.last_contours, self.last_scan_numbers, angles=self.last_angles, delaunay_alpha=self.delaunay_alpha)
                    self.visualize_model(model)
                    volume_mm3 = self.builder.volume
                    volume_ml = volume_mm3 / VOLUME_DIVIDER
                    self.volume_label.setText(f"Объём: {volume_mm3:.3f} мм³ ({volume_ml:.4f} мл)")
            except ValueError:
                show_error("Неверный формат масштаба, количества точек или alpha. Используются значения по умолчанию.")
                self._reset_scales(scale_x_input, scale_y_input)

    def _reset_scales(self, scale_x_input, scale_y_input):
        """Reset scale inputs to default values."""
        if self.builder is None:
            return
        scale_x_input.setText(f"{self.builder.IMAGE_WIDTH / self.builder.REAL_WIDTH:.2f}")
        scale_y_input.setText(f"{self.builder.IMAGE_HEIGHT / self.builder.REAL_HEIGHT:.2f}")

    def visualize_model(self, mesh):
        try:
            if self.plotter is not None:
                self.plotter.clear()
            else:
                self.plotter = self.vtk_widget

            # Облако точек
            points = self.builder.points
            self.plotter.add_points(points, color='lightgreen', point_size=2, render_points_as_spheres=True, name='points')

            # Delaunay mesh: заливка и wireframe
            self.plotter.add_mesh(mesh, color='darkred', opacity=0.1, name='fill', lighting=False)
            self.plotter.add_mesh(mesh, color='white', opacity=0.3, style='wireframe', name='wire', line_width=0.75)

            self.plotter.set_background((0.1, 0.1, 0.15))
            self.plotter.reset_camera()
            axes = pv.AxesAssembly(label_color='white', label_size=12)
            self.plotter.add_orientation_widget(axes)
            self.plotter.update()
        except Exception as e:
            show_error(f"Ошибка визуализации: {str(e)}")

    def copy_volume(self, event):
        QtWidgets.QApplication.clipboard().setText(self.volume_label.text())

    def open_single_image(self):
        """Открывает отдельное изображение и показывает распознанный контур через hsv-маску."""
        try:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if not file_path:
                return
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                show_error(f"Не удалось загрузить изображение: {file_path}")
                return
            # Обработка изображения для выделения контура
            contour = ImageProcessor.process_image(img, saturation_threshold=self.processor.saturation_threshold)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > (self.processor.saturation_threshold * 255)
            mask = mask.astype(np.uint8) * 255
            # Визуализация результата
            vis_img = img.copy()
            if contour is not None:
                cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
            # Добавим маску как\ альфа-канал для наглядности
            if vis_img.shape[2] == 3:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2BGRA)
            vis_img[:, :, 3] = mask
            # Покажем результат в отдельном окне PyQt
            self.show_image_dialog(vis_img, title=f"Контур на изображении: {Path(file_path).name}")
        except Exception as e:
            show_error(f"Ошибка обработки изображения: {str(e)}")

    def show_image_dialog(self, img, title="Результат"):
        """Показывает изображение в отдельном диалоговом окне."""
        try:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(title)
            vbox = QtWidgets.QVBoxLayout(dlg)

            # Convert to RGB and ensure C-contiguous for QImage
            # Input 'img' is expected to be BGR from cv2.imdecode
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb_contiguous = np.ascontiguousarray(img_rgb)

            h, w, ch = img_rgb_contiguous.shape
            bytes_per_line = ch * w

            # Keep a reference to the bytes object to prevent premature garbage collection
            # Store it as an attribute of the dialog instance
            dlg._image_buffer_for_qimage = img_rgb_contiguous.tobytes()

            qimg = QtGui.QImage(dlg._image_buffer_for_qimage, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)

            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            label.setScaledContents(True) # Make image scale to label size if label is resized
            vbox.addWidget(label)

            dlg.resize(min(w, 800), min(h, 600))
            dlg.exec()
        except Exception as e:
            show_error(f"Ошибка отображения изображения: {str(e)}")

    def open_debug_viewer(self):
        """Открывает отладочный просмотрщик изображений."""
        try:
            if not hasattr(self, 'last_images') or not self.last_images:
                show_error("Нет данных для отладки. Сначала выберите папку с изображениями.")
                return
                
            # Передаем данные в отладочный просмотрщик
            self.debug_viewer.set_data(
                self.last_images,
                self.last_scan_numbers,
                self.last_contours,
                self.last_image_files
            )
            
            # Показываем окно
            self.debug_viewer.show()
            self.debug_viewer.raise_()
            self.debug_viewer.activateWindow()
            
        except Exception as e:
            show_error(f"Ошибка открытия отладочного просмотрщика: {str(e)}")
            logging.error(f"Ошибка открытия отладочного просмотрщика: {str(e)}", exc_info=True)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(True) # Ensure application quits when last window is closed
    window = MainWindow()
    window.show()
    app.exec()