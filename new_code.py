import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QAction
from PyQt6 import QtGui
import pyvista as pv
from pyvistaqt import QtInteractor
import logging

# Настройка логирования
logging.basicConfig(filename='scan_processor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def show_error(message: str, level: str = 'critical'):
    """
    Универсальная функция для отображения ошибок и логирования.
    level: 'critical' или 'warning'
    """
    if level == 'critical':
        logging.error(message)
    else:
        logging.warning(message)
    try:
        app = QApplication.instance()
        if app is not None:
            if level == 'critical':
                QMessageBox.critical(None, "Ошибка", message)
            else:
                QMessageBox.warning(None, "Внимание", message)
    except Exception:
        pass

# === КОНСТАНТЫ ===
ROI_PERCENTAGE = 0.05
MIN_CONTOUR_AREA = 4
CONFIDENCE_THRESHOLD = 0.7
TARGET_NORM_SIZE = (20, 32)  # (ширина, высота)
MORPH_KERNEL_MAX_SIZE = 50
DEFAULT_REAL_WIDTH = 10.0  # mm
DEFAULT_REAL_HEIGHT = 2.0  # mm
SCAN_NUMBER_MIN = 1
SCAN_NUMBER_MAX = 18
DELAUNAY_ALPHA = 250.0
CONTOUR_APPROX_RATE = 0.0025
CONTOUR_MIN_POINTS = 4
VOLUME_DIVIDER = 1000.0

class DataReader:
    def __init__(self, directory, templates_dir="templates"):
        self.directory = Path(directory)
        self.templates_dir = Path(templates_dir)
        self.images = []
        self.scan_numbers = []
        self.templates = None
        self.ROI_PERCENTAGE = ROI_PERCENTAGE
        self.MIN_CONTOUR_AREA = MIN_CONTOUR_AREA
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.TARGET_NORM_SIZE = TARGET_NORM_SIZE  # (ширина, высота)
        self.image_shape = None  # (height, width, channels)
        self._load_templates()

    def _imread_unicode(self, path):
        """Читает изображение с поддержкой Unicode."""
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

    def _load_and_crop_roi(self, image_path):
        """Загружает изображение и вырезает ROI."""
        img_bgr = self._imread_unicode(image_path)
        if img_bgr is None:
            show_error(f"Не удалось загрузить изображение: {image_path}")
            return None, None, None, None

        h_orig, w_orig = img_bgr.shape[:2]
        roi_size = int(min(h_orig, w_orig) * self.ROI_PERCENTAGE)
        if roi_size < 5:
            show_error(f"ROI слишком мал для изображения: {image_path}")
            return None, None, None, None
        x1, y1 = 0, 0
        x2, y2 = roi_size, roi_size

        roi_bgr = img_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            show_error(f"Пустой ROI для изображения: {image_path}")
            return None, None, None, None
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return roi_bgr, roi_gray, img_bgr, (x1, y1, x2, y2)

    def _find_number_bbox(self, gray_roi):
        """Находит ограничивающий прямоугольник для числа."""
        try:
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                show_error("Контуры числа не найдены", level='warning')
                return None

            valid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_CONTOUR_AREA]
            if not valid_contours:
                show_error("Валидные контуры числа не найдены", level='warning')
                return None

            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            for c in valid_contours:
                x, y, w, h = cv2.boundingRect(c)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

            return (min_x, min_y, max_x - min_x, max_y - min_y)
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
            normalized_number = cv2.resize(number_img, self.TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
            return normalized_number
        except Exception as e:
            show_error(f"Ошибка нормализации числа: {str(e)}")
            return None

    def _load_templates(self):
        """Загружает нормализованные шаблоны чисел."""
        self.templates = {}
        for i in range(1, 19):
            template_path = self.templates_dir / f"{i}.png"
            template_bgr = self._imread_unicode(template_path)
            if template_bgr is None:
                show_error(f"Шаблон для числа {i} не найден: {template_path}", level='warning')
                continue
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            bbox = self._find_number_bbox(template_gray)
            normalized_template = self._extract_and_normalize_number(template_gray, bbox)
            if normalized_template is not None:
                self.templates[i] = normalized_template
        if not self.templates:
            show_error("Не удалось загрузить ни один шаблон")
            self.templates = None

    def _recognize_number(self, normalized_number_img):
        """Распознаёт число с помощью шаблонов."""
        if normalized_number_img is None or self.templates is None or not self.templates:
            show_error("Нет нормализованного изображения числа или шаблонов", level='warning')
            return None, None

        best_match_number = None
        best_match_value = -1.0

        for num, template in self.templates.items():
            if normalized_number_img.shape != template.shape:
                continue
            res = cv2.matchTemplate(normalized_number_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_match_value:
                best_match_value = max_val
                best_match_number = num

        if best_match_value < self.CONFIDENCE_THRESHOLD:
            show_error(f"Число не распознано, уверенность: {best_match_value:.2f}", level='warning')
            return None, best_match_value
        return best_match_number, best_match_value

    def _extract_number(self, roi_gray):
        """Извлекает номер из изображения."""
        try:
            bbox = self._find_number_bbox(roi_gray)
            normalized_number = self._extract_and_normalize_number(roi_gray, bbox)
            number, confidence = self._recognize_number(normalized_number)
            if number is not None:
                logging.info(f"Распознано число {number} с уверенностью {confidence:.2f}")
            return number
        except Exception as e:
            show_error(f"Ошибка извлечения номера: {str(e)}")
            return None

    def read_images(self):
        """Читает изображения и извлекает номера сканов. Проверяет одинаковое разрешение."""
        try:
            self.images = []
            self.scan_numbers = []
            seen_numbers = set()  # Для отслеживания уникальных номеров
            image_shape = None
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                for file in self.directory.glob(ext):
                    _, roi_gray, img, _ = self._load_and_crop_roi(file)
                    number = self._extract_number(roi_gray)
                    if number is not None and 1 <= number <= 18:
                        if number in seen_numbers:
                            show_error(f"Обнаружен дубликат номера скана {number} в файле: {file}", level='warning')
                            continue
                        if img is None:
                            continue
                        if image_shape is None:
                            image_shape = img.shape
                        elif img.shape != image_shape:
                            show_error(f"Обнаружено изображение с другим разрешением: {file} (ожидалось {image_shape}, получено {img.shape})")
                            raise ValueError(f"Обнаружено изображение с другим разрешением: {file} (ожидалось {image_shape}, получено {img.shape})")
                        self.images.append(img)
                        self.scan_numbers.append(number)
                        seen_numbers.add(number)
                    else:
                        show_error(f"Неверный номер скана в файле: {file}", level='warning')

            missing = set(range(1, 19)) - set(self.scan_numbers)
            if missing:
                show_error(f"Отсутствуют сканы: {missing}", level='warning')
            if not self.images:
                show_error("Не найдено ни одного валидного изображения")
                raise ValueError("Не найдено ни одного валидного изображения")
            self.image_shape = image_shape
            return self.images, self.scan_numbers, image_shape
        except Exception as e:
            show_error(f"Ошибка чтения данных: {str(e)}")
            raise


def process_image_func(img, saturation_threshold=0.0, approximation_rate=0.0025):
    """Выделяет контур из изображения (функция верхнего уровня для multiprocessing)."""
    import cv2
    import numpy as np
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > (saturation_threshold * 255)
        h, w = mask.shape
        kernel_size = min(MORPH_KERNEL_MAX_SIZE, h, w)
        if kernel_size < 1:
            show_error("Размер ядра морфологии слишком мал")
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 4:
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


class ImageProcessor:
    def __init__(self, saturation_threshold=0.0):
        self.saturation_threshold = saturation_threshold

    def process_image(self, img, approximation_rate=0.0025):
        """Выделяет контур из изображения."""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > (self.saturation_threshold * 255)
            h, w = mask.shape
            kernel_size = min(MORPH_KERNEL_MAX_SIZE, h, w)
            if kernel_size < 1:
                show_error("Размер ядра морфологии слишком мал")
                return None
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if len(contour) < 4:
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
    def __init__(self, image_width, image_height, real_width=DEFAULT_REAL_WIDTH, real_height=DEFAULT_REAL_HEIGHT):
        self.points = None  # numpy array (N, 3)
        self.mesh = None    # pyvista.PolyData
        self.volume = 0.0
        self.IMAGE_WIDTH = image_width  # pixels
        self.IMAGE_HEIGHT = image_height  # pixels
        self.REAL_WIDTH = real_width   # mm
        self.REAL_HEIGHT = real_height   # mm
        self.scale_x = self.IMAGE_WIDTH / self.REAL_WIDTH if self.REAL_WIDTH != 0 else 1.0  # pixels per mm
        self.scale_y = self.IMAGE_HEIGHT / self.REAL_HEIGHT if self.REAL_HEIGHT != 0 else 1.0 # pixels per mm
        logging.info(f"Default scales calculated: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm")

    def set_scale(self, scale_x, scale_y):
        self.scale_x = scale_x if scale_x > 0 else (self.IMAGE_WIDTH / self.REAL_WIDTH if self.REAL_WIDTH != 0 else 1.0)
        self.scale_y = scale_y if scale_y > 0 else (self.IMAGE_HEIGHT / self.REAL_HEIGHT if self.REAL_HEIGHT != 0 else 1.0)
        logging.info(f"Scale set: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm")

    TARGET_MIN_ANGLE_STEP = 4.0 # degrees

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
            logging.error(f"Error in _perform_interpolation_step: {str(e)}", exc_info=True)
            show_error(f"Ошибка интерполяции контуров: {str(e)}")
            return existing_contours, existing_angles

    def _interpolate_contour(self, c1, c2, t):
        try:
            n_points = min(len(c1), len(c2))
            c1 = cv2.approxPolyDP(c1, 1.0, True)[:n_points]
            c2 = cv2.approxPolyDP(c2, 1.0, True)[:n_points]
            return (c1 * (1 - t) + c2 * t).astype(np.int32)
        except Exception as e:
            logging.error(f"Ошибка интерполяции контура: {str(e)}")
            return c1

    def build_model(self, contours, scan_numbers, angles=None, center=None):
        try:
            if not contours or not scan_numbers:
                show_error("Нет валидных контуров или номеров сканов")
                raise ValueError("Нет валидных контуров или номеров сканов")
            if angles is None:
                N = len(scan_numbers)
                if N == 0:
                    show_error("Нет валидных сканов для вычисления углов")
                    raise ValueError("Нет валидных сканов для вычисления углов")
                if 360 % N != 0:
                    show_error(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом.")
                    raise ValueError(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом.")
                initial_angle_step = 360.0 / N # Use float division
                initial_angles = [i * initial_angle_step for i in range(N)]
            else:
                initial_angles = angles[:] # Use a copy to avoid modifying original list

            # Инициализируем текущие данные для итеративной интерполяции
            current_contours = contours[:] # Копируем начальный список
            current_angles = initial_angles[:] # Копируем начальный список

            # Определяем начальный эффективный угловой шаг
            if len(current_angles) > 1:
                current_effective_angle_step = current_angles[1] - current_angles[0]
            else:
                # Если только один контур, считаем, что это полный круг (360 градусов)
                current_effective_angle_step = 360.0

            # Цикл для выполнения шагов интерполяции до тех пор, пока не будет достигнуто целевое угловое разрешение
            interpolation_iteration = 0
            while current_effective_angle_step > self.TARGET_MIN_ANGLE_STEP:
                interpolation_iteration += 1
                logging.info(f"Interpolation iteration {interpolation_iteration}: Current effective angle step {current_effective_angle_step:.2f} degrees. Target: {self.TARGET_MIN_ANGLE_STEP:.2f} degrees.")

                new_contours, new_angles = self._perform_interpolation_step(current_contours, current_angles)
                
                # Проверяем, произошла ли интерполяция (т.е. стало ли больше контуров)
                if len(new_contours) == len(current_contours):
                    logging.info("Interpolation step did not increase contour density. Stopping interpolation.")
                    break # Останавливаем, если новые контуры не были добавлены, или если функция вернула оригинал из-за ошибок

                current_contours = new_contours
                current_angles = new_angles
                
                # Пересчитываем текущий эффективный угловой шаг на основе новых углов
                if len(current_angles) > 1:
                    current_effective_angle_step = current_angles[1] - current_angles[0]
                else:
                    # Если каким-то образом после интерполяции остался только один контур, прерываем, чтобы избежать бесконечного цикла
                    logging.warning("Only one contour remains after interpolation. Stopping.")
                    break
                
                # Добавляем защиту от бесконечных циклов из-за неточностей плавающей точки
                if interpolation_iteration > 10: # Максимум 10 итераций (например, 360 -> 180 -> ... -> ~0.35)
                    logging.warning("Too many interpolation iterations, potential infinite loop detected. Stopping.")
                    break

            # После цикла интерполяции присваиваем окончательные контуры и углы
            contours = current_contours
            angles = current_angles
            # Перегенерируем фиктивные scan_numbers для логгирования/совместимости, если необходимо
            scan_numbers = list(range(1, len(contours) + 1))
            logging.info(f"Final number of contours after interpolation: {len(contours)}, with angular step: {current_effective_angle_step:.2f} degrees.")

            if center is None:
                center = (self.IMAGE_WIDTH // 2, self.IMAGE_HEIGHT // 2)
            points_list = []
            for i, contour in enumerate(contours):
                angle_rad = angles[i] * np.pi / 180
                for point in contour:
                    x, y = point[0]
                    x_centered = (x - center[0]) / self.scale_x
                    y_centered = (y - center[1]) / self.scale_y
                    x_3d = x_centered * np.cos(angle_rad)
                    y_3d = y_centered
                    z_3d = x_centered * np.sin(angle_rad)
                    points_list.append([x_3d, y_3d, z_3d])
            points = np.array(points_list)
            unique_points = np.unique(points, axis=0)
            print(f"Всего точек: {points.shape[0]}, уникальных: {unique_points.shape[0]}")
            logging.info(f"Всего точек: {points.shape[0]}, уникальных: {unique_points.shape[0]}")
            if points.shape[0] < 4:
                show_error(f"Недостаточно точек для триангуляции: {points.shape[0]}")
                raise ValueError(f"Недостаточно точек для триангуляции: {points.shape[0]}")
            self.points = points
            try:
                cloud = pv.PolyData(points)
                mesh = cloud.delaunay_3d(alpha=DELAUNAY_ALPHA)
                surf = mesh.extract_geometry()
                print(f"Full mesh: {surf.n_faces} faces, volume: {surf.volume:.2f}")
                logging.info(f"Full mesh: {surf.n_faces} faces, volume: {surf.volume:.2f}")
                self.mesh = surf
                self.volume = surf.volume
                return surf
            except Exception as e:
                show_error(f"Ошибка Delaunay триангуляции: {str(e)}")
                raise ValueError(f"Ошибка построения 3D-модели: {str(e)}")
        except Exception as e:
            show_error(f"Ошибка построения модели: {str(e)}")
            raise


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Scan Processor (PyVista)")
        self.setGeometry(100, 100, 800, 600)
        self.reader = DataReader('.')
        self.processor = ImageProcessor()
        self.builder = None
        self.is_point_mode = True
        self.plotter = None
        self.point_actor = None
        self.surface_actor = None
        self.progress_bar = None
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

    def _find_optimal_alpha(self, contours, scan_numbers, angles):
        logging.info("Starting optimal DELAUNAY_ALPHA search using adaptive approach...")

        global DELAUNAY_ALPHA # Declare intent to modify the global variable
        original_delaunay_alpha = DELAUNAY_ALPHA # Store original value

        try:
            # Phase 1: Binary Search for the smallest alpha that yields a manifold mesh
            low_alpha_bound = 1.0
            high_alpha_bound = 500.0 # Increased upper bound to give more room for search
            best_manifold_alpha_in_range = None
            
            binary_search_iterations = 15 # Number of binary search steps

            # Total steps for progress bar will be binary_search_iterations + linear_scan_steps
            linear_scan_steps = 15
            total_progress_steps = binary_search_iterations + linear_scan_steps

            self.progress_bar.setFormat("Поиск Alpha (фаза 1/2): %p%")
            self.progress_bar.setMaximum(total_progress_steps)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            QtWidgets.QApplication.processEvents()

            logging.info(f"Phase 1: Binary search for smallest manifold alpha. Range: [{low_alpha_bound}, {high_alpha_bound}]")
            
            for i in range(binary_search_iterations):
                current_alpha = (low_alpha_bound + high_alpha_bound) / 2
                DELAUNAY_ALPHA = current_alpha # Temporarily modify the global DELAUNAY_ALPHA

                try:
                    if self.builder is None:
                        logging.error("ModelBuilder is not initialized in _find_optimal_alpha.")
                        raise ValueError("ModelBuilder not initialized.")
                    
                    model = self.builder.build_model(contours, scan_numbers, angles=angles)
                    is_manifold = model.is_manifold
                    volume = model.volume
                    
                    logging.info(f"  Binary Alpha={current_alpha:.2f}: Manifold={is_manifold}, Volume={volume:.2f}")

                    if is_manifold and volume > 0:
                        best_manifold_alpha_in_range = current_alpha
                        high_alpha_bound = current_alpha # Try smaller alpha
                    else:
                        low_alpha_bound = current_alpha # Need larger alpha

                except ValueError as e:
                    logging.warning(f"  Binary Search: Failed to build model for Alpha={current_alpha:.2f}: {e}")
                    low_alpha_bound = current_alpha # If failed, it's likely too small or invalid, so move up
                except Exception as e:
                    logging.error(f"  Binary Search: Unexpected error for Alpha={current_alpha:.2f}: {e}", exc_info=True)
                    low_alpha_bound = current_alpha # If failed, treat as non-manifold and move up

                self.progress_bar.setValue(i + 1)
                QtWidgets.QApplication.processEvents()

            if best_manifold_alpha_in_range is None:
                logging.warning("Binary search failed to find a manifold alpha. Reverting to linear scan over original range.")
                # If binary search fails, fall back to a wide linear scan for robustness
                alpha_values_for_linear_scan = np.linspace(10.0, 500.0, 50) # Original full linear scan
                best_manifold_alpha_in_range = original_delaunay_alpha # Fallback
                
            else:
                logging.info(f"Phase 1 complete. Smallest manifold alpha found (approx): {best_manifold_alpha_in_range:.2f}")
                # Phase 2: Linear scan in a refined window around the best_manifold_alpha_in_range
                # Adjust the range based on the found alpha
                search_start = max(10.0, best_manifold_alpha_in_range - 50) # Go slightly below the found alpha
                search_end = best_manifold_alpha_in_range + 200 # Go significantly above to find stable region

                # Ensure search_end is reasonable if best_manifold_alpha_in_range is already high
                if search_end > 1000.0:
                    search_end = 1000.0
                if search_start >= search_end: # Prevent inverted range, ensure minimum width
                    search_start = max(10.0, search_end - 100)
                
                alpha_values_for_linear_scan = np.linspace(search_start, search_end, linear_scan_steps)
                logging.info(f"Phase 2: Linear scan in refined range: [{search_start:.2f}, {search_end:.2f}] with {linear_scan_steps} steps.")

            candidate_alpha_results = [] # Store results for manifold meshes (potential candidates)
            
            self.progress_bar.setFormat("Поиск Alpha (фаза 2/2): %p%")
            # Starting value is binary_search_iterations to continue from where phase 1 left off.

            for idx, current_alpha in enumerate(alpha_values_for_linear_scan):
                DELAUNAY_ALPHA = current_alpha # Temporarily modify the global DELAUNAY_ALPHA

                try:
                    model = self.builder.build_model(contours, scan_numbers, angles=angles)
                    volume = model.volume
                    n_faces = model.n_faces
                    is_manifold = model.is_manifold

                    logging.info(f"  Linear Alpha={current_alpha:.2f}: Volume={volume:.2f} mm³, Faces={n_faces}, Manifold={is_manifold}")

                    if is_manifold and volume > 0:
                        candidate_alpha_results.append({
                            'alpha': current_alpha,
                            'volume': volume
                        })

                except ValueError as e:
                    logging.warning(f"  Linear Scan: Failed to build model for Alpha={current_alpha:.2f}: {e}")
                except Exception as e:
                    logging.error(f"  Linear Scan: Unexpected error for Alpha={current_alpha:.2f}: {e}", exc_info=True)

                self.progress_bar.setValue(binary_search_iterations + idx + 1)
                QtWidgets.QApplication.processEvents()

        finally:
            DELAUNAY_ALPHA = original_delaunay_alpha # Always restore original DELAUNAY_ALPHA
            self.progress_bar.setVisible(False) # Hide progress bar after search

        if not candidate_alpha_results:
            logging.warning("No suitable alpha value found to create a manifold mesh in refined scan. Using default DELAUNAY_ALPHA.")
            return original_delaunay_alpha

        # Analyze collected candidate results (only manifold ones)
        volumes = [r['volume'] for r in candidate_alpha_results]
        if not volumes: # This check is technically redundant if candidate_alpha_results is not empty
            logging.warning("No valid volumes found among manifold candidates. Using default DELAUNAY_ALPHA.")
            return original_delaunay_alpha

        median_volume = np.median(volumes)
        best_alpha_candidate = None
        min_diff_from_median = float('inf')

        # Find the alpha that produces a manifold mesh whose volume is closest to the median
        for r in candidate_alpha_results:
            current_diff = abs(r['volume'] - median_volume)
            if current_diff < min_diff_from_median:
                min_diff_from_median = current_diff
                best_alpha_candidate = r['alpha']

        logging.info(f"Optimal DELAUNAY_ALPHA selected: {best_alpha_candidate:.2f} (median volume of candidates: {median_volume:.2f} mm³)")
        return best_alpha_candidate

    def select_folder(self):
        try:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
            if folder:
                self.reader.directory = Path(folder)
                images, scan_numbers, image_shape = self.reader.read_images()
                if image_shape is None:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось определить разрешение изображений")
                    raise ValueError("Не удалось определить разрешение изображений")
                image_height, image_width = image_shape[:2]
                N = len(scan_numbers)
                if N == 0:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", "Не найдено ни одного валидного изображения для построения модели")
                    raise ValueError("Не найдено ни одного валидного изображения для построения модели")
                if 360 % N != 0:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом. Попробуйте другое количество кадров.")
                    raise ValueError(f"Количество кадров ({N}) не делит 360 нацело. Угол между срезами должен быть целым числом. Попробуйте другое количество кадров.")
                angle = 360 // N
                angles = [i * angle for i in range(N)]
                self.builder = ModelBuilder(image_width, image_height)

                # --- Прогресс-бар для обработки контуров ---
                self.progress_bar.setFormat("Обработка изображений: %p%")
                self.progress_bar.setVisible(True)
                self.progress_bar.setMaximum(len(images))
                self.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()

                # Для совместимости с multiprocessing и прогресс-баром используем последовательную обработку (можно доработать для параллелизма через callback)
                contours = []
                for idx, img in enumerate(images):
                    contour = process_image_func(img)
                    if contour is not None:
                        contours.append(contour)
                    self.progress_bar.setValue(idx + 1)
                    QtWidgets.QApplication.processEvents()

                self.progress_bar.setVisible(False) # Hide after contour processing

                if not contours:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось извлечь ни одного контура")
                    raise ValueError("Не удалось извлечь ни одного контура")

                # Сохраняем данные для возможного перестроения модели (например, при изменении масштаба)
                self.last_contours = contours
                self.last_scan_numbers = scan_numbers
                self.last_angles = angles

                # --- Find optimal DELAUNAY_ALPHA ---
                optimal_alpha = self._find_optimal_alpha(contours, scan_numbers, angles)
                show_error(f"Optimal DELAUNAY_ALPHA: {optimal_alpha:.2f}")

                # Set the global DELAUNAY_ALPHA to the optimal value for the final model build
                global DELAUNAY_ALPHA
                DELAUNAY_ALPHA = optimal_alpha

                # Build the final model with the optimal alpha
                model = self.builder.build_model(contours, scan_numbers, angles=angles)
                self.visualize_model(model)
                volume_mm3 = self.builder.volume
                volume_ml = volume_mm3 / VOLUME_DIVIDER
                self.volume_label.setText(f"Объём: {volume_mm3:.4f} мм³ ({volume_ml:.5f} мл)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка обработки: {str(e)}")
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)

    def open_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки масштаба")
        layout = QtWidgets.QVBoxLayout(dialog)

        if self.builder is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Модель не инициализирована. Сначала выберите папку с изображениями.")
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
                self.builder.set_scale(scale_x, scale_y)
                # Перестраиваем модель с новыми масштабами
                if hasattr(self, 'last_contours') and hasattr(self, 'last_scan_numbers') and hasattr(self, 'last_angles'):
                    model = self.builder.build_model(self.last_contours, self.last_scan_numbers, angles=self.last_angles)
                    self.visualize_model(model)
                    volume_mm3 = self.builder.volume
                    volume_ml = volume_mm3 / VOLUME_DIVIDER
                    self.volume_label.setText(f"Объём: {volume_mm3:.3f} мм³ ({volume_ml:.4f} мл)")
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Ошибка",
                                            "Неверный формат масштаба. Используются значения по умолчанию.")
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
            # Удаляем все actor-ы перед добавлением новых
            if self.point_actor:
                self.plotter.remove_actor(self.point_actor)
                self.point_actor = None
            if self.surface_actor:
                self.plotter.remove_actor(self.surface_actor)
                self.surface_actor = None
            # Облако точек
            points = self.builder.points
            self.point_actor = self.plotter.add_points(points, color='white', point_size=2, render_points_as_spheres=True, name='points')
            # Delaunay mesh: заливка и wireframe
            fill_actor = self.plotter.add_mesh(mesh, color='darkred', opacity=0.1, name='fill', lighting=False)
            wire_actor = self.plotter.add_mesh(mesh, color='white', opacity=0.3, style='wireframe', name='wire', line_width=0.75)
            self.plotter.set_background((0.1, 0.1, 0.15))
            self.plotter.reset_camera()
            axes = pv.AxesAssembly(label_color='white', label_size=12)
            self.plotter.add_orientation_widget(axes)
            self.plotter.update()
        except Exception as e:
            logging.error(f"Ошибка визуализации: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка визуализации: {str(e)}")

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
                QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение: {file_path}")
                return
            # Обработка изображения для выделения контура
            contour = process_image_func(img)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > (self.processor.saturation_threshold * 255)
            mask = mask.astype(np.uint8) * 255
            # Визуализация результата
            vis_img = img.copy()
            if contour is not None:
                cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
            # Добавим маску как альфа-канал для наглядности
            if vis_img.shape[2] == 3:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2BGRA)
            vis_img[:, :, 3] = mask
            # Покажем результат в отдельном окне PyQt
            self.show_image_dialog(vis_img, title=f"Контур на изображении: {os.path.basename(file_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка обработки изображения: {str(e)}")

    def show_image_dialog(self, img, title="Результат"):
        """Показывает изображение в отдельном диалоговом окне."""
        try:
            # Преобразуем изображение для Qt
            if img.shape[2] == 4:
                fmt = QtGui.QImage.Format.Format_RGBA8888
            else:
                fmt = QtGui.QImage.Format.Format_RGB888
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            qimg = QtGui.QImage(img.data, w, h, img.strides[0], fmt)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(title)
            vbox = QtWidgets.QVBoxLayout(dlg)
            vbox.addWidget(label)
            dlg.resize(min(w, 800), min(h, 600))
            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка отображения изображения: {str(e)}")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()