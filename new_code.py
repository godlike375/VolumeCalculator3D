import logging
from dataclasses import dataclass
from pathlib import Path
import cv2, numpy as np, pyvista as pv, matplotlib.pyplot as plt
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QMessageBox
from pyvistaqt import QtInteractor

# === КОНСТАНТЫ ЛОГИРОВАНИЯ ===
LOG_FILENAME, LOG_FORMAT, LOG_LEVEL = "scan_processor.log", "%(asctime)s - %(levelname)s - %(message)s", logging.INFO
ROI_PERCENTAGE, MIN_CONTOUR_AREA, CONFIDENCE_THRESHOLD = 0.025, 4, 0.7
TARGET_NORM_SIZE, MORPH_KERNEL_MAX_SIZE, TEMPLATES_DIR = (20, 32), 2, "templates"
DEFAULT_REAL_WIDTH, DEFAULT_REAL_HEIGHT = 10.0, 2.0
SCAN_NUMBER_MIN, SCAN_NUMBER_MAX, DELAUNAY_ALPHA = 1, 99, 50.0
CONTOUR_APPROX_RATE, CONTOUR_MIN_POINTS, VOLUME_DIVIDER = 0.000225, 4, 1000.0
TARGET_MIN_ANGLE_STEP, OUTLIER_THRESHOLD_PERCENT_DEFAULT, OUTLIER_ABSOLUTE_THRESHOLD_MM = 10, 0.25, 0.1

# Настройка логирования
# Для совместимости с Python < 3.9, открываем файл явно с utf-8 кодировкой
log_file_handler = logging.FileHandler(LOG_FILENAME, mode="a", encoding="utf-8")
logging.basicConfig(
    handlers=[log_file_handler], level=LOG_LEVEL, format=LOG_FORMAT
)  # Removed encoding argument, added handlers


def show_error(message: str, level: str = "critical"):
    """Универсальная функция для отображения ошибок и логирования."""
    log_func, dialog_func, console_prefix = {"critical": (logging.error, QMessageBox.critical, "CRITICAL ERROR"), "warning": (logging.warning, QMessageBox.warning, "WARNING")}[level]
    log_func(message)
    app = QApplication.instance()
    (dialog_func(None, "Ошибка" if level == "critical" else "Внимание", message) if app else print(f"{console_prefix}: {message}"))


# === УТИЛИТЫ ДЛЯ РАБОТЫ С ИЗОБРАЖЕНИЯМИ ===
# --- Исправление длинных однострочников в DataReader ---
def resample_contour(contour: np.ndarray, n_points: int = 100) -> np.ndarray:
    pts = contour.squeeze()
    if len(pts.shape) == 1:
        pts = pts[None, :]
    if pts.shape[0] < 2:
        return contour
    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    dists = np.insert(dists, 0, 0)
    cumulative = np.cumsum(dists)
    total_length = cumulative[-1]
    if total_length == 0:
        return contour
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
    return np.array(new_pts, dtype=np.int32).reshape(-1, 1, 2)


# === DATACLASS ДЛЯ НАСТРОЕК ===
@dataclass
class ModelSettings:
    real_width: float = 10.0
    real_height: float = 2.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    resample_points: int = 100
    delaunay_alpha: float = 50.0
    outlier_detection_enabled: bool = True
    outlier_threshold_percent: float = 0.25
    outlier_absolute_threshold_mm: float = 0.1


class DataReader:
    def __init__(self, directory, templates_dir=TEMPLATES_DIR):
        self.directory, self.templates_dir = Path(directory), Path(templates_dir)
        self.images, self.scan_numbers, self.digit_templates, self.image_shape = [], [], {}, None
        self._load_digit_templates_from_templates()

    def _imread_unicode(self, path):
        try:
            with open(path, "rb") as f:
                img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: raise ValueError(f"Не удалось декодировать изображение: {path}")
            return img
        except Exception as e:
            logging.error(f"Ошибка чтения изображения {path}: {str(e)}"); return None

    def _load_digit_templates_from_templates(self):
        self.digit_templates = {}
        for digit in range(10):
            template_path = self.templates_dir / f"{digit}.png"
            template_bgr = self._imread_unicode(template_path)
            if template_bgr is None:
                continue
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            bbox = self._find_number_bbox(template_gray)
            norm = self._extract_and_normalize_number(template_gray, bbox)
            if norm is not None:
                self.digit_templates[digit] = norm
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
            valid = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
            if not valid: show_error("Валидные контуры числа не найдены", level="warning"); return None
            all_points = np.vstack(valid).squeeze(); min_x, min_y = all_points.min(axis=0); max_x, max_y = all_points.max(axis=0)
            return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        except Exception as e:
            show_error(f"Ошибка поиска контура числа: {str(e)}"); return None

    def _extract_and_normalize_number(self, gray_roi, bbox):
        """Извлекает и нормализует изображение числа."""
        try:
            if bbox is None:
                return None
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                show_error(f"Некорректные размеры bbox: w={w}, h={h}")
                return None
            number_img = gray_roi[y : y + h, x : x + w]
            if number_img.size == 0:
                show_error(f"Пустое изображение числа после вырезки bbox")
                return None
            return cv2.resize(number_img, TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
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
        _, thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        bboxes = sorted(bboxes, key=lambda b: b[0])
        digit_imgs = []
        for x, y, w, h in bboxes:
            digit = roi_gray[y : y + h, x : x + w]
            norm_digit = cv2.resize(digit, TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
            digit_imgs.append(norm_digit)
        return digit_imgs

    def _recognize_digit(self, digit_img):
        if not (digit_img is not None and self.digit_templates):
            show_error("Нет изображения цифры или шаблонов", level="warning")
            return None, None
        best_digit, best_val = None, -1.0
        for digit, template in self.digit_templates.items():
            if digit_img.shape != template.shape:
                continue
            res = cv2.matchTemplate(digit_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_digit = digit
        if best_val < CONFIDENCE_THRESHOLD:
            show_error(f"Цифра не распознана, уверенность: {best_val:.2f}", level="warning")
            return None, best_val
        return best_digit, best_val

    def _extract_number(self, roi_gray):
        try:
            digit_imgs = self._extract_digits_from_roi(roi_gray)
            digits = []
            for digit_img in digit_imgs:
                digit, _ = self._recognize_digit(digit_img)
                if digit is not None:
                    digits.append(digit)
            if not digits:
                return None
            return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
        except Exception as e:
            show_error(f"Ошибка извлечения номера: {str(e)}")
            return None

    def read_images(self):
        try:
            self.images, self.scan_numbers, self.image_files, seen_numbers, image_shape, all_image_paths = [], [], [], set(), None, []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]: all_image_paths.extend(self.directory.glob(ext))
            all_image_paths = sorted(all_image_paths, key=lambda p: p.name)
            for file_path in all_image_paths:
                img_bgr = self._imread_unicode(file_path)
                if img_bgr is None: continue
                roi_bgr, roi_gray = self._find_number_roi(img_bgr)
                number = self._extract_number(roi_gray)
                if number is None or not (SCAN_NUMBER_MIN <= number <= SCAN_NUMBER_MAX): show_error(f"Неверный номер скана в файле: {file_path.name}", level="warning"); continue
                if number in seen_numbers: show_error(f"Обнаружен дубликат номера скана {number} в файле: {file_path.name}", level="warning"); continue
                if image_shape is None: image_shape = img_bgr.shape
                elif img_bgr.shape != image_shape: show_error(f"Обнаружено изображение с другим разрешением: {file_path.name} (ожидалось {image_shape}, получено {img_bgr.shape})"); raise ValueError(f"Обнаружено изображение с другим разрешением: {file_path.name} (ожидалось {image_shape}, получено {img_bgr.shape})")
                self.images.append(img_bgr); self.scan_numbers.append(number); self.image_files.append(file_path); seen_numbers.add(number)
            sorted_data = sorted(zip(self.scan_numbers, self.images, self.image_files))
            self.scan_numbers, self.images, self.image_files = [item[0] for item in sorted_data], [item[1] for item in sorted_data], [item[2] for item in sorted_data]
            missing = set(range(min(self.scan_numbers), max(self.scan_numbers) + 1)) - set(self.scan_numbers) if self.scan_numbers else set()
            if missing: show_error(f"Отсутствуют сканы: {sorted(list(missing))}", level="warning")
            if not self.images: show_error("Не найдено ни одного валидного изображения"); raise ValueError("Не найдено ни одного валидного изображения")
            self.image_shape = image_shape
            return self.images, self.scan_numbers, image_shape
        except Exception as e:
            show_error(f"Ошибка чтения данных: {str(e)}")
            raise


class ImageProcessor:
    def __init__(self, saturation_threshold=24, hue_max=62):
        self.saturation_threshold = saturation_threshold
        self.hue_max = hue_max
    @staticmethod
    def process_image(img: np.ndarray, approximation_rate: float = CONTOUR_APPROX_RATE, saturation_threshold: float = 0.0, hue_max: int = 40) -> np.ndarray:
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = (hsv[:, :, 1] > saturation_threshold) & (hsv[:, :, 0] <= hue_max)
            h, w = mask.shape
            kernel_size = min(MORPH_KERNEL_MAX_SIZE, h, w)
            if kernel_size < 1:
                show_error("Размер ядра морфологии слишком мал")
                return None
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            for _ in range(3):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if len(contour) < CONTOUR_MIN_POINTS:
                    show_error("Контур слишком мал", level="warning")
                    return None
                arclen = cv2.arcLength(contour, True)
                epsilon = arclen * approximation_rate
                approx = cv2.approxPolyDP(contour, epsilon, True)
                return resample_contour(approx)
            show_error("Контур не найден", level="warning")
            return None
        except Exception as e:
            show_error(f"Ошибка обработки изображения: {str(e)}")
            return None


class ModelBuilder:
    def __init__(self, image_width, image_height, real_width=DEFAULT_REAL_WIDTH, real_height=DEFAULT_REAL_HEIGHT, n_resample_points=100):
        self.settings = ModelSettings(real_width=real_width, real_height=real_height, scale_x=image_width / real_width if real_width else 1.0, scale_y=image_height / real_height if real_height else 1.0, resample_points=n_resample_points)
        self.points, self.mesh, self.volume = None, None, 0.0
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.REAL_WIDTH, self.REAL_HEIGHT = image_width, image_height, real_width, real_height
        self.scale_x, self.scale_y, self.n_resample_points = self.settings.scale_x, self.settings.scale_y, n_resample_points
        self.outlier_detection_enabled, self.outlier_threshold_percent, self.outlier_absolute_threshold_mm = True, OUTLIER_THRESHOLD_PERCENT_DEFAULT, OUTLIER_ABSOLUTE_THRESHOLD_MM
        logging.info(f"Default scales calculated: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm")
    def set_scale(self, scale_x: float, scale_y: float):
        self.scale_x = scale_x if scale_x > 0 else self.IMAGE_WIDTH / self.REAL_WIDTH
        self.scale_y = scale_y if scale_y > 0 else self.IMAGE_HEIGHT / self.REAL_HEIGHT
        self.settings.scale_x, self.settings.scale_y = self.scale_x, self.scale_y
        logging.info(f"Scale set: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm")
    def _resample_contour(self, contour, n_points=None):
        n = n_points or getattr(self, 'n_resample_points', 100)
        return resample_contour(contour, n)
    def _interpolate_contour(self, c1, c2, t):
        try:
            if c1 is None or c2 is None or c1.shape != c2.shape:
                logging.error(f"Не удалось ресэмплировать контуры для интерполяции. c1r shape: {c1.shape if c1 is not None else 'None'}, c2r shape: {c2.shape if c2 is not None else 'None'}")
                return c1 if c1 is not None else c1
            interp = c1.astype(np.float32) * (1 - t) + c2.astype(np.float32) * t
            return interp.astype(np.int32)
        except Exception as e:
            logging.error(f"Ошибка интерполяции контура: {str(e)}", exc_info=True)
            try:
                return self._resample_contour(c1)
            except Exception:
                return c1
    def _filter_outliers(self, contours_3d_points_list, angles, center):
        if not self.outlier_detection_enabled or len(contours_3d_points_list) < 3:
            return contours_3d_points_list
        filtered = []
        n = 0
        for i, curr in enumerate(contours_3d_points_list):
            prev = contours_3d_points_list[(i-1)%len(contours_3d_points_list)]
            nxt = contours_3d_points_list[(i+1)%len(contours_3d_points_list)]
            if prev.shape[0] != curr.shape[0] or nxt.shape[0] != curr.shape[0]:
                filtered.append(curr)
                continue
            r_curr = np.abs(curr[:,0])
            r_prev = np.abs(prev[:,0])
            r_next = np.abs(nxt[:,0])
            mod = np.copy(curr)
            for j in range(curr.shape[0]):
                avg = (r_prev[j] + r_next[j]) / 2.0
                if (avg < self.outlier_absolute_threshold_mm * 2 and abs(r_curr[j] - avg) > self.outlier_absolute_threshold_mm) or (avg > 1e-9 and abs(r_curr[j] - avg) / avg > self.outlier_threshold_percent):
                    mod[j] = (prev[j] + nxt[j]) / 2.0
                    n += 1
            filtered.append(mod)
        logging.info(f"Фильтрация выбросов завершена. Всего скорректировано точек: {n}" if n else "Фильтрация выбросов завершена. Выбросов не обнаружено.")
        return filtered

    def build_model(
        self, contours, scan_numbers, angles=None, center=None, delaunay_alpha=None
    ):
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
                if 180 % N != 0:
                    show_error(
                        f"Количество кадров ({N}) не делит 180 нацело. Угол между срезами должен быть целым числом."
                    )
                    raise ValueError(
                        f"Количество кадров ({N}) не делит 180 нацело. Угол между срезами должен быть целым числом."
                    )
                initial_angle_step = 180.0 / N
                initial_angles = [i * initial_angle_step for i in range(N)]
            else:
                initial_angles = angles[:]

            # Интерполяция контуров
            current_contours = contours[:]
            current_angles = initial_angles[:]

            if len(current_angles) > 1:
                current_effective_angle_step = (
                    sorted(current_angles)[1] - sorted(current_angles)[0]
                )
            else:
                current_effective_angle_step = 180

            # Цикл интерполяции

            for i, existing_contour in enumerate(current_contours):
                current_contours[i] = self._resample_contour(current_contours[i])

            interpolation_iteration = 0
            while current_effective_angle_step > TARGET_MIN_ANGLE_STEP + 1e-9:
                interpolation_iteration += 1
                logging.info(
                    f"Interpolation iteration {interpolation_iteration}: Current effective angle step {current_effective_angle_step:.2f} degrees. Target: {TARGET_MIN_ANGLE_STEP:.2f} degrees."
                )

                new_contours, new_angles = self._perform_interpolation_step(
                    current_contours, current_angles
                )

                if len(new_contours) == len(current_contours):
                    logging.info(
                        "Interpolation step did not increase contour density. Stopping interpolation."
                    )
                    break

                current_contours = new_contours
                current_angles = new_angles

                if len(current_angles) > 1:
                    current_effective_angle_step = (
                        sorted(current_angles)[1] - sorted(current_angles)[0]
                    )
                else:
                    logging.warning(
                        "Only one contour remains after interpolation. Stopping."
                    )
                    break

                if interpolation_iteration > 15:
                    logging.warning(
                        "Too many interpolation iterations, potential infinite loop detected. Stopping."
                    )
                    break

            contours = current_contours
            angles = current_angles
            scan_numbers = list(range(1, len(contours) + 1))
            logging.info(
                f"Final number of contours after interpolation: {len(contours)}, with angular step: {current_effective_angle_step:.2f} degrees."
            )

            # --- ФИЛЬТРАЦИЯ ВЫБРОСОВ ---
            # Преобразуем список контуров (которые сейчас являются 2D) в список 3D-точек для фильтрации
            # Затем обратно в 2D-контуры для остальной логики.
            # Это может показаться неэффективным, но пока это самый простой способ интегрировать 3D-фильтр
            # до окончательного построения 3D-модели.
            if center is None:
                center = (self.IMAGE_WIDTH // 2, self.IMAGE_HEIGHT // 2)

            contours_as_3d_points = []
            for contour in contours:
                current_contour_3d_points = []
                for point in contour:
                    x, y = point[0]
                    x_physical = (x - center[0]) / self.scale_x
                    y_physical = (center[1] - y) / self.scale_y
                    # На этом этапе z_3d = 0, поскольку мы еще не применили поворот
                    # x_3d и z_3d будут пересчитаны после фильтрации
                    current_contour_3d_points.append(
                        [x_physical, y_physical, 0.0]
                    )  # Z пока 0
                contours_as_3d_points.append(np.array(current_contour_3d_points))

            # Применяем фильтрацию выбросов
            filtered_3d_points_per_contour = self._filter_outliers(
                contours_as_3d_points, angles, center
            )  # Передаем center

            # Преобразуем отфильтрованные 3D-точки обратно в 2D-контуры (для совместимости с существующим кодом)
            # Однако, фактически мы будем использовать эти 3D-точки для построения финальной модели
            # Эта часть кода немного избыточна, но пока она соответствует текущей структуре.
            filtered_contours = []
            for contour_3d_points in filtered_3d_points_per_contour:
                # Здесь мы берем только X и Y компоненты (по сути, 2D проекцию на плоскость XY)
                # и приводим их обратно к пикселям для корректного формата контура OpenCV
                temp_contour_2d = np.array(
                    [
                        [
                            int(p[0] * self.scale_x + center[0]),
                            int(center[1] - p[1] * self.scale_y),
                        ]
                        for p in contour_3d_points
                    ],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                filtered_contours.append(temp_contour_2d)

            # Важно: для построения 3D модели мы будем использовать filtered_3d_points_per_contour
            # а не filtered_contours.

            # Создание 3D точек
            # Center уже определен выше.
            # points_list теперь формируем из filtered_3d_points_per_contour
            points_list = []
            for i, contour_3d_points_array in enumerate(filtered_3d_points_per_contour):
                angle_rad = angles[i] * np.pi / 180
                for p_physical in contour_3d_points_array:
                    x_physical = p_physical[0]
                    y_physical = p_physical[1]

                    x_3d = x_physical * np.cos(angle_rad)
                    y_3d = y_physical
                    z_3d = x_physical * np.sin(angle_rad)

                    points_list.append([x_3d, y_3d, z_3d])

            points = np.array(points_list)
            unique_points = np.unique(points, axis=0)
            logging.info(
                f"Всего точек: {points.shape[0]}, уникальных: {unique_points.shape[0]}"
            )
            if points.shape[0] < 4:
                show_error(f"Недостаточно точек для триангуляции: {points.shape[0]}")
                raise ValueError(
                    f"Недостаточно точек для триангуляции: {points.shape[0]}"
                )

            self.points = points

            # НОВОЕ: Сохраняем отфильтрованные 3D-срезы и интерполированные углы для детальной отрисовки контуров
            self.individual_contour_3d_points = filtered_3d_points_per_contour
            self.angles = angles

            # Создание 3D модели
            current_delaunay_alpha = (
                delaunay_alpha if delaunay_alpha is not None else DELAUNAY_ALPHA
            )
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
        self.images, self.scan_numbers, self.contours, self.image_files = [], [], [], []
        self.current_index = 0
        self.init_ui()
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        info_layout = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel("Нет данных")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        info_layout.addWidget(self.info_label)
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
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        layout.addWidget(self.view)
        details_layout = QtWidgets.QHBoxLayout()
        self.details_label = QtWidgets.QLabel("Детали распознавания:")
        self.details_label.setStyleSheet("font-size: 12px;")
        details_layout.addWidget(self.details_label)
        layout.addLayout(details_layout)
        help_layout = QtWidgets.QHBoxLayout()
        help_text = ("Подсказки: ←/→ навигация, колесико мыши - масштаб, 0 - сброс масштаба, "
                     "Esc - закрыть, перетаскивание мышью - перемещение")
        help_label = QtWidgets.QLabel(help_text)
        help_label.setStyleSheet("font-size: 10px; color: gray; font-style: italic;")
        help_layout.addWidget(help_label)
        layout.addLayout(help_layout)
    def set_data(self, images, scan_numbers, contours, image_files):
        self.images, self.scan_numbers, self.contours, self.image_files = images, scan_numbers, contours, image_files
        self.current_index = 0
        if self.images:
            self.update_navigation_buttons()
            self.show_current_image()
        else:
            self.info_label.setText("Нет данных для отображения")
    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.images) - 1)
    def show_current_image(self):
        if not self.images or self.current_index >= len(self.images):
            return
        img = self.images[self.current_index]
        scan_number = self.scan_numbers[self.current_index] if self.current_index < len(self.scan_numbers) else "N/A"
        contour = self.contours[self.current_index] if self.current_index < len(self.contours) else None
        file_name = self.image_files[self.current_index] if self.current_index < len(self.image_files) else "unknown"
        display_img = img.copy()
        h, w = img.shape[:2]
        roi_h, roi_w = int(h * 0.2), int(w * 0.2)
        cv2.rectangle(display_img, (0, 0), (roi_w, roi_h), (0, 255, 255), 2)
        cv2.putText(display_img, "ROI", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if contour is not None:
            n_contours = len(self.contours)
            if n_contours > 0:
                colors = plt.cm.hsv(np.linspace(0, 1, n_contours, endpoint=False))[:, :3]
                color = tuple(int(c * 255) for c in colors[self.current_index % n_contours])
            else:
                color = (0, 255, 0)
            cv2.drawContours(display_img, [contour], -1, color, 2)
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = display_img_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(display_img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.info_label.setText(f"Изображение {self.current_index + 1}/{len(self.images)} | "
                                f"Номер скана: {scan_number} | "
                                f"Файл: {Path(file_name).name}")
        details = []
        if contour is not None:
            details.append(f"Контур найден: {len(contour)} точек")
            area = cv2.contourArea(contour)
            details.append(f"Площадь контура: {area:.1f} пикселей²")
            perimeter = cv2.arcLength(contour, True)
            details.append(f"Периметр: {perimeter:.1f} пикселей")
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                details.append(f"Компактность: {compactness:.3f}")
        else:
            details.append("Контур не найден")
        if scan_number != "N/A":
            details.append(f"Распознан номер: {scan_number}")
        else:
            details.append("Номер не распознан")
        details.append(f"Размер: {w}x{h} пикселей")
        self.details_label.setText(" | ".join(details))
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_navigation_buttons()
            self.show_current_image()
    def show_next(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.update_navigation_buttons()
            self.show_current_image()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene and not self.scene.sceneRect().isEmpty():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.show_previous()
        elif event.key() == Qt.Key.Key_Right:
            self.show_next()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_0:
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            super().keyPressEvent(event)
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.view.scale(1.1, 1.1)
        else:
            self.view.scale(0.9, 0.9)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Scan Processor (PyVista)")
        self.setGeometry(100, 100, 800, 600)
        self.reader = DataReader(".")
        self.processor = ImageProcessor()
        self.builder = None
        self.plotter = None
        self.progress_bar = None
        self.resample_points = 250
        self.delaunay_alpha = DELAUNAY_ALPHA
        self.debug_viewer = DebugViewer(self)
        self.init_ui()
    def init_ui(self):
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
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Файл")
        open_action = QAction("&Открыть папку...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Выбрать папку с изображениями сканов")
        open_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        debug_action = QAction("&Отладочный просмотрщик...", self)
        debug_action.setShortcut("Ctrl+D")
        debug_action.setStatusTip("Открыть отладочный просмотрщик для проверки распознавания")
        debug_action.triggered.connect(self.open_debug_viewer)
        file_menu.addAction(debug_action)
        file_menu.addSeparator()
        exit_action = QAction("&Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Выйти из приложения")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        tools_menu = menubar.addMenu("&Инструменты")
        settings_action = QAction("&Настройки масштаба...", self)
        settings_action.setShortcut("Ctrl+S")
        settings_action.setStatusTip("Настроить масштаб модели")
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)
        help_menu = menubar.addMenu("&Справка")
        about_action = QAction("&О программе", self)
        about_action.setStatusTip("Информация о программе")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    def show_about(self):
        QtWidgets.QMessageBox.about(self, "О программе",
            "3D Scan Processor\n\n"
            "Программа для обработки 3D сканов и расчета объема\n"
            "Использует PyVista для 3D визуализации\n\n"
            "Версия 1.0")
    def _set_progress(self, visible: bool, maximum: int = 100, value: int = 0, text: str = ""):
        if self.progress_bar is not None:
            self.progress_bar.setVisible(visible)
            if text:
                self.progress_bar.setFormat(text)
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
            QtWidgets.QApplication.processEvents()
    def _find_optimal_alpha(
        self, contours, scan_numbers, angles, initial_delaunay_alpha
    ):
        logging.info(
            "Starting optimal DELAUNAY_ALPHA search using adaptive approach..."
        )
        try:
            low_alpha_bound = 10.0
            high_alpha_bound = 250.0
            best_manifold_alpha_in_range = None
            binary_search_iterations = 1
            linear_scan_steps = 1
            total_progress_steps = binary_search_iterations + linear_scan_steps
            self._set_progress(
                True, total_progress_steps, 0, "Поиск Alpha (фаза 1/2): %p%"
            )
            for i in range(binary_search_iterations):
                current_alpha = (low_alpha_bound + high_alpha_bound) / 2
                try:
                    if self.builder is None:
                        logging.error(
                            "ModelBuilder is not initialized in _find_optimal_alpha."
                        )
                        raise ValueError("ModelBuilder not initialized.")
                    model = self.builder.build_model(
                        contours,
                        scan_numbers,
                        angles=angles,
                        delaunay_alpha=current_alpha,
                    )
                    is_manifold = model.is_manifold
                    volume = model.volume
                    logging.info(
                        f"  Binary Alpha={current_alpha:.2f}: Manifold={is_manifold}, Volume={volume:.2f}"
                    )
                    if is_manifold and volume > 0:
                        best_manifold_alpha_in_range = current_alpha
                        high_alpha_bound = current_alpha
                    else:
                        low_alpha_bound = current_alpha
                except ValueError as e:
                    logging.warning(
                        f"  Binary Search: Failed to build model for Alpha={current_alpha:.2f}: {e}"
                    )
                    low_alpha_bound = current_alpha
                except Exception as e:
                    logging.error(
                        f"  Binary Search: Unexpected error for Alpha={current_alpha:.2f}: {e}",
                        exc_info=True,
                    )
                    low_alpha_bound = current_alpha
                self._set_progress(True, total_progress_steps, i + 1)
            if best_manifold_alpha_in_range is None:
                logging.warning(
                    f"Binary search failed to find a manifold alpha in range [1.0, 1000.0]. Reverting to a broader linear scan starting from initial_delaunay_alpha."
                )
                search_start = max(1.0, initial_delaunay_alpha - 100)
                search_end = initial_delaunay_alpha + 300
                alpha_values_for_linear_scan = np.linspace(
                    search_start, search_end, linear_scan_steps
                )
                best_manifold_alpha_in_range = initial_delaunay_alpha
            else:
                logging.info(
                    f"Phase 1 complete. Smallest manifold alpha found (approx): {best_manifold_alpha_in_range:.2f}"
                )
                search_start = max(1.0, best_manifold_alpha_in_range - 50)
                search_end = best_manifold_alpha_in_range + 200
                if search_end > 1000.0:
                    search_end = 1000.0
                if search_start >= search_end:
                    search_start = max(1.0, search_end - 100)
                alpha_values_for_linear_scan = np.linspace(
                    search_start, search_end, linear_scan_steps
                )
                logging.info(
                    f"Phase 2: Linear scan in refined range: [{search_start:.2f}, {search_end:.2f}] with {linear_scan_steps} steps."
                )
            candidate_alpha_results = []
            self._set_progress(
                True,
                total_progress_steps,
                binary_search_iterations,
                "Поиск Alpha (фаза 2/2): %p%",
            )
            for idx, current_alpha in enumerate(alpha_values_for_linear_scan):
                try:
                    model = self.builder.build_model(
                        contours,
                        scan_numbers,
                        angles=angles,
                        delaunay_alpha=current_alpha,
                    )
                    volume = model.volume
                    n_faces = model.n_faces
                    is_manifold = model.is_manifold
                    logging.info(
                        f"  Linear Alpha={current_alpha:.2f}: Volume={volume:.2f} мм³, Faces={n_faces}, Manifold={is_manifold}"
                    )
                    if is_manifold and volume > 0:
                        candidate_alpha_results.append(
                            {"alpha": current_alpha, "volume": volume}
                        )
                except ValueError as e:
                    logging.warning(
                        f"  Linear Scan: Failed to build model for Alpha={current_alpha:.2f}: {e}"
                    )
                except Exception as e:
                    logging.error(
                        f"  Linear Scan: Unexpected error for Alpha={current_alpha:.2f}: {e}",
                        exc_info=True,
                    )
                self._set_progress(
                    True, total_progress_steps, binary_search_iterations + idx + 1
                )
        finally:
            self._set_progress(False)
        if not candidate_alpha_results:
            logging.warning(
                "No suitable alpha value found to create a manifold mesh in refined scan. Using default DELAUNAY_ALPHA."
            )
            return initial_delaunay_alpha
        volumes = [r["volume"] for r in candidate_alpha_results]
        if not volumes:
            logging.warning(
                "No valid volumes found among manifold candidates. Using default DELAUNAY_ALPHA."
            )
            return initial_delaunay_alpha
        median_volume = np.median(volumes)
        best_alpha_candidate = None
        min_diff_from_median = float("inf")
        for r in candidate_alpha_results:
            current_diff = abs(r["volume"] - median_volume)
            if current_diff < min_diff_from_median:
                min_diff_from_median = current_diff
                best_alpha_candidate = r["alpha"]
        logging.info(
            f"Optimal DELAUNAY_ALPHA selected: {best_alpha_candidate:.2f} (median volume of candidates: {median_volume:.2f} мм³)"
        )
        return best_alpha_candidate
    def select_folder(self):
        try:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Выберите папку с изображениями"
            )
            if folder:
                self.reader.directory = Path(folder)
                images, scan_numbers, image_shape = self.reader.read_images()
                if image_shape is None:
                    show_error("Не удалось определить разрешение изображений")
                    raise ValueError("Не удалось определить разрешение изображений")
                image_height, image_width = image_shape[:2]
                N = len(scan_numbers)
                if N == 0:
                    show_error(
                        "Не найдено ни одного валидного изображения для построения модели"
                    )
                    raise ValueError(
                        "Не найдено ни одного валидного изображения для построения модели"
                    )
                if 180 % N != 0:
                    show_error(
                        f"Количество кадров ({N}) не делит 180 нацело. Угол между срезами должен быть целым числом. Попробуйте другое количество кадров."
                    )
                    raise ValueError(
                        f"Количество кадров ({N}) не делит 180 нацело. Угол между срезами должен быть целым числом. Попробуйте другое количество кадров."
                    )
                angle = 180 // N
                angles = [i * angle for i in range(N)]
                self.builder = ModelBuilder(image_width, image_height)
                self._set_progress(True, len(images), 0, "Обработка изображений: %p%")
                contours = self._extract_contours(images)
                self._set_progress(False)
                if not contours:
                    show_error("Не удалось извлечь ни одного контура")
                    raise ValueError("Не удалось извлечь ни одного контура")
                self.last_contours = contours
                self.last_scan_numbers = scan_numbers
                self.last_angles = angles
                self.last_images = images
                self.last_image_files = [f.name for f in self.reader.image_files]
                optimal_alpha = self._find_optimal_alpha(
                    contours, scan_numbers, angles, DELAUNAY_ALPHA
                )
                show_error(f"Optimal DELAUNAY_ALPHA: {optimal_alpha:.2f}")
                model = self.builder.build_model(
                    contours, scan_numbers, angles=angles, delaunay_alpha=optimal_alpha
                )
                self.visualize_model(model)
                volume_mm3 = self.builder.volume
                volume_ml = volume_mm3 / VOLUME_DIVIDER
                self.volume_label.setText(
                    f"Объём: {volume_mm3:.4f} мм³ ({volume_ml:.5f} мл)"
                )
        except Exception as e:
            show_error(f"Ошибка обработки: {str(e)}")
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)
    def _extract_contours(self, images: list) -> list:
        """Извлекает контуры из списка изображений."""
        contours = []
        for idx, img in enumerate(images):
            contour = ImageProcessor.process_image(
                img,
                saturation_threshold=self.processor.saturation_threshold,
                hue_max=self.processor.hue_max,
            )
            if contour is not None:
                contours.append(contour)
            self._set_progress(True, len(images), idx + 1)
        return contours
    def open_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки масштаба")
        layout = QtWidgets.QVBoxLayout(dialog)
        if self.builder is None:
            show_error(
                "Модель не инициализирована. Сначала выберите папку с изображениями."
            )
            return
        info_label = QtWidgets.QLabel(
            f"Размеры изображения: {self.builder.IMAGE_WIDTH}x{self.builder.IMAGE_HEIGHT} пикселей\n"
            f"Реальные размеры: {self.builder.REAL_WIDTH}x{self.builder.REAL_HEIGHT} мм"
        )
        layout.addWidget(info_label)
        current_scale_label = QtWidgets.QLabel(
            f"Текущий масштаб:\n"
            f"X,Z: {self.builder.scale_x:.2f} пикселей/мм\n"
            f"Y: {self.builder.scale_y:.2f} пикселей/мм"
        )
        layout.addWidget(current_scale_label)
        layout.addWidget(QtWidgets.QLabel("Масштаб по X и Z (пикселей на мм):"))
        scale_x_input = QtWidgets.QLineEdit()
        scale_x_input.setText(f"{self.builder.scale_x:.2f}")
        layout.addWidget(scale_x_input)
        layout.addWidget(QtWidgets.QLabel("Масштаб по Y (пикселей на мм):"))
        scale_y_input = QtWidgets.QLineEdit()
        scale_y_input.setText(f"{self.builder.scale_y:.2f}")
        layout.addWidget(scale_y_input)
        layout.addWidget(QtWidgets.QLabel("Количество точек на контуре (ремэппинг):"))
        resample_points_input = QtWidgets.QLineEdit()
        resample_points_input.setText(str(self.resample_points))
        layout.addWidget(resample_points_input)
        layout.addWidget(QtWidgets.QLabel("DELAUNAY ALPHA (параметр триангуляции):"))
        delaunay_alpha_input = QtWidgets.QLineEdit()
        delaunay_alpha_input.setText(str(self.delaunay_alpha))
        layout.addWidget(delaunay_alpha_input)
        layout.addWidget(
            QtWidgets.QLabel("Максимальный оттенок (Hue Max для фильтрации): ")
        )
        hue_max_input = QtWidgets.QLineEdit()
        hue_max_input.setText(str(self.processor.hue_max))
        layout.addWidget(hue_max_input)
        filter_outliers_group_box = QtWidgets.QGroupBox("Настройки фильтрации выбросов")
        filter_outliers_layout = QtWidgets.QVBoxLayout(filter_outliers_group_box)
        self.outlier_enabled_checkbox = QtWidgets.QCheckBox(
            "Включить фильтрацию выбросов"
        )
        self.outlier_enabled_checkbox.setChecked(self.builder.outlier_detection_enabled)
        filter_outliers_layout.addWidget(self.outlier_enabled_checkbox)
        filter_outliers_layout.addWidget(
            QtWidgets.QLabel("Порог относительного отклонения (%): ")
        )
        self.outlier_threshold_percent_input = QtWidgets.QLineEdit()
        self.outlier_threshold_percent_input.setText(
            f"{self.builder.outlier_threshold_percent:.3f}"
        )
        filter_outliers_layout.addWidget(self.outlier_threshold_percent_input)
        filter_outliers_layout.addWidget(
            QtWidgets.QLabel("Порог абсолютного отклонения (мм): ")
        )
        self.outlier_absolute_threshold_mm_input = QtWidgets.QLineEdit()
        self.outlier_absolute_threshold_mm_input.setText(
            f"{self.builder.outlier_absolute_threshold_mm:.3f}"
        )
        filter_outliers_layout.addWidget(self.outlier_absolute_threshold_mm_input)
        layout.addWidget(filter_outliers_group_box)
        button_layout = QtWidgets.QHBoxLayout()
        reset_button = QtWidgets.QPushButton("Сбросить")
        reset_button.clicked.connect(
            lambda: self._reset_scales(scale_x_input, scale_y_input)
        )
        button_layout.addWidget(reset_button)
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                scale_x = float(scale_x_input.text())
                scale_y = float(scale_y_input.text())
                resample_points = int(resample_points_input.text())
                delaunay_alpha = float(delaunay_alpha_input.text())
                hue_max = int(hue_max_input.text())
                outlier_detection_enabled = self.outlier_enabled_checkbox.isChecked()
                outlier_threshold_percent = float(
                    self.outlier_threshold_percent_input.text()
                )
                outlier_absolute_threshold_mm = float(
                    self.outlier_absolute_threshold_mm_input.text()
                )
                if resample_points < 4:
                    resample_points = 4
                if delaunay_alpha < 1.0:
                    delaunay_alpha = 1.0
                if outlier_threshold_percent < 0:
                    outlier_threshold_percent = 0
                if outlier_absolute_threshold_mm < 0:
                    outlier_absolute_threshold_mm = 0
                self.resample_points = resample_points
                self.delaunay_alpha = delaunay_alpha
                self.builder.set_scale(scale_x, scale_y)
                self.processor.hue_max = hue_max
                self.builder.outlier_detection_enabled = outlier_detection_enabled
                self.builder.outlier_threshold_percent = outlier_threshold_percent
                self.builder.outlier_absolute_threshold_mm = (
                    outlier_absolute_threshold_mm
                )
                if (
                    hasattr(self, "last_contours")
                    and hasattr(self, "last_scan_numbers")
                    and hasattr(self, "last_angles")
                ):
                    self.builder.n_resample_points = self.resample_points
                    model = self.builder.build_model(
                        self.last_contours,
                        self.last_scan_numbers,
                        angles=self.last_angles,
                        delaunay_alpha=self.delaunay_alpha,
                    )
                    self.visualize_model(model)
                    volume_mm3 = self.builder.volume
                    volume_ml = volume_mm3 / VOLUME_DIVIDER
                    self.volume_label.setText(
                        f"Объём: {volume_mm3:.3f} мм³ ({volume_ml:.4f} мл)"
                    )
            except ValueError:
                show_error(
                    "Неверный формат масштаба, количества точек, alpha или порогов фильтрации. Используются предыдущие значения."
                )
    def _reset_scales(self, scale_x_input, scale_y_input):
        """Reset scale inputs to default values."""
        if self.builder is None:
            return
        scale_x_input.setText(
            f"{self.builder.IMAGE_WIDTH / self.builder.REAL_WIDTH:.2f}"
        )
        scale_y_input.setText(
            f"{self.builder.IMAGE_HEIGHT / self.builder.REAL_HEIGHT:.2f}"
        )
    def visualize_model(self, mesh):
        try:
            if self.plotter is not None:
                self.plotter.clear()
            else:
                self.plotter = self.vtk_widget
            points = self.builder.points
            self.plotter.add_points(
                points,
                color="lightgreen",
                point_size=2,
                render_points_as_spheres=True,
                name="points",
            )
            self.plotter.add_mesh(
                mesh, color="darkred", opacity=0.1, name="fill", lighting=False
            )
            self.plotter.add_mesh(
                mesh,
                color="white",
                opacity=0.3,
                style="wireframe",
                name="wire",
                line_width=0.75,
            )
            if (
                self.builder is not None
                and hasattr(self.builder, "individual_contour_3d_points")
                and hasattr(self.builder, "angles")
            ):
                individual_contour_3d_points_list = (
                    self.builder.individual_contour_3d_points
                )
                angles = self.builder.angles
                n_contours_for_coloring = len(individual_contour_3d_points_list)
                colors = plt.cm.hsv(
                    np.linspace(0, 1, n_contours_for_coloring, endpoint=False)
                )[:, :3]
                for i, group_points_raw in enumerate(individual_contour_3d_points_list):
                    if i >= len(angles):
                        logging.warning(
                            f"Angle for contour {i} out of bounds. Skipping colored line for this slice."
                        )
                        continue
                    angle_rad = angles[i] * np.pi / 180
                    x_3d_rotated = group_points_raw[:, 0] * np.cos(angle_rad)
                    y_3d_rotated = group_points_raw[:, 1]
                    z_3d_rotated = group_points_raw[:, 0] * np.sin(angle_rad)
                    rotated_group_points = np.stack(
                        [x_3d_rotated, y_3d_rotated, z_3d_rotated], axis=-1
                    )
                    if len(rotated_group_points) > 1:
                        segments = []
                        num_points_in_contour = len(rotated_group_points)
                        for j in range(num_points_in_contour):
                            p1_idx = j
                            p2_idx = (
                                j + 1
                            ) % num_points_in_contour
                            segments.extend([2, p1_idx, p2_idx])
                        lines = np.array(segments, dtype=np.int32)
                        poly = pv.PolyData(rotated_group_points)
                        poly.lines = lines
                        color = tuple((colors[i] * 255).astype(int))
                        self.plotter.add_mesh(
                            poly, color=color, line_width=3, name=f"scanline_{i}"
                        )
            self.plotter.set_background((0.1, 0.1, 0.15))
            self.plotter.reset_camera()
            axes = pv.AxesAssembly(label_color="white", label_size=12)
            self.plotter.add_orientation_widget(axes)
            self.plotter.update()
        except Exception as e:
            show_error(f"Ошибка визуализации: {str(e)}")
    def copy_volume(self, event):
        QtWidgets.QApplication.clipboard().setText(self.volume_label.text())
    def open_debug_viewer(self):
        if not hasattr(self, "last_images") or not self.last_images:
            show_error("Нет данных для отладки. Сначала выберите папку с изображениями.")
            return
        self.debug_viewer.set_data(self.last_images, self.last_scan_numbers, self.last_contours, self.last_image_files)
        self.debug_viewer.show()
        self.debug_viewer.raise_()
        self.debug_viewer.activateWindow()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(
        True
    )  # Ensure application quits when last window is closed
    window = MainWindow()
    window.show()
    app.exec()
