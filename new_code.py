
import logging
import traceback
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QGraphicsScene
from PyQt6.QtWidgets import QGraphicsView, QMessageBox
from pyvistaqt import QtInteractor
import json
from typing import List, Tuple, Dict, Any

LOG_FILENAME = "scan_processor.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
log_file_handler = logging.FileHandler(LOG_FILENAME, mode="a", encoding="utf-8")
logging.basicConfig(handlers=[log_file_handler], level=LOG_LEVEL, format=LOG_FORMAT)


class ErrorDialog(QtWidgets.QDialog):
    def __init__(self, message, log_text=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ошибка")
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        if log_text:
            log_box = QtWidgets.QTextEdit()
            log_box.setReadOnly(True)
            log_box.setPlainText(log_text)
            log_box.setMinimumHeight(150)
            layout.addWidget(log_box)
        btn = QtWidgets.QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# --- Обновленная функция show_error ---
def show_error(message: str, level: str = "critical", exc: Exception = None, tb: str = None):
    mapping = {"critical": (logging.error, QMessageBox.critical, "CRITICAL ERROR"),
               "warning": (logging.warning, QMessageBox.warning, "INFO"), # Changed QMessageBox.warning to QMessageBox.information for INFO level
               "info": (logging.info, QMessageBox.information, "INFO")}
    log_func, dialog_func, prefix = mapping[level]
    log_func(message)
    app = QApplication.instance()
    log_text = ""
    if exc or tb:
        if tb is None and exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log_text = tb
    if app:
        dlg = ErrorDialog(message, log_text)
        dlg.exec()
    else:
        print(f"{prefix}: {message}")
        if log_text:
            print(log_text)


def resample_contour(contour: np.ndarray, n_points: int = 150) -> np.ndarray:
    pts = contour.squeeze()
    if len(pts.shape) == 1:
        pts = pts[None, :]
    if pts.shape[0] < 2:
        return contour
    closed_pts = np.vstack([pts, pts[0]])
    seg_vecs = np.diff(closed_pts, axis=0)
    seg_lens = np.sqrt(np.sum(seg_vecs ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 0:
        return contour.astype(np.float32).reshape(-1, 1, 2)
    # Семплируем без дублирования начальной точки
    targets = np.linspace(0.0, total, int(n_points), endpoint=False)
    new_pts = []
    for t in targets:
        # Найти сегмент, куда попадает t
        idx = int(np.searchsorted(cum, t, side='right') - 1)
        idx = max(0, min(idx, len(seg_lens) - 1))
        t0, t1 = cum[idx], cum[idx + 1]
        p0 = closed_pts[idx]
        p1 = closed_pts[idx + 1]
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        new_pts.append((1.0 - alpha) * p0 + alpha * p1)
    # Возвращаем с плавающей точкой для сохранения точности; приводите к int там, где это требуется OpenCV
    return np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)


class Settings:
    MIN_CONTOUR_AREA = 4
    CONFIDENCE_THRESHOLD = 0.01
    TARGET_NORM_SIZE = (20, 32)
    MORPH_KERNEL_MAX_SIZE = 10
    DEFAULT_REAL_WIDTH = 10.0
    DEFAULT_REAL_HEIGHT = 2.0
    SCAN_NUMBER_MIN = 1
    SCAN_NUMBER_MAX = 99
    CONTOUR_APPROX_RATE = 0.0001
    VOLUME_DIVIDER = 1000.0
    TEMPLATES_DIR = "templates"
    ARROW_HSV_LOWER = [28, 16, 165]
    ARROW_HSV_UPPER = [36, 255, 255]
    ARROW_MIN_CONTOUR_AREA = 20
    ARROW_SYMMETRY_EPSILON = 1e-2
    NUMBER_BIN_THRESH = 200
    NUMBER_ROI_PERCENT = 0.05
    ARROW_ROI_PERCENT = 0.1
    MORPH_DILATE_ITER = 2
    MORPH_ERODE_KERNEL_DIV_W = 9.0
    MORPH_ERODE_KERNEL_DIV_H = 5.5
    MORPH_ERODE_EXTRA_ITERATIONS = 6
    CONTOUR_HSV_LOWER = [12, 33, 170]
    CONTOUR_HSV_UPPER = [30, 255, 212]
    SATURATION_THRESHOLD = 24
    ARROW_MIN_CONTOUR_POINTS = 10
    MIN_CONTOUR_POINTS = 4
    MIN_ANGLE_BETWEEN_CONTOURS = 1.25
    EASING_STRENGTH = 0.5  # 0.0=линейная, 1.0=кубическая
    VOXEL_SIZE_MM = 0.025
    VOXEL_MAX_VOXELS = 10000000
    VOXEL_REQUIRED_FRACTION = 0.7
    VOXEL_MASK_DILATE_PX = 1
    MIN_CONTOUR_AREA_PERCENTAGE = 0.1
    BORDER_TOLERANCE_PX = 1

    @classmethod
    def save(cls, path="settings.json"):
        d = {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v) and isinstance(v, (int, float, bool, str, list, tuple))}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path="settings.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            for k, v in d.items():
                if hasattr(cls, k):
                    setattr(cls, k, v)
        except Exception:
            pass


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(500)
        self.inputs = {}
        tab_widget = QtWidgets.QTabWidget(self)
        tabs = [
            ("Масштаб и основное", [
                ("DEFAULT_REAL_WIDTH", "Реальная ширина (мм)", float),
                ("DEFAULT_REAL_HEIGHT", "Реальная высота (мм)", float),
                ("SCAN_NUMBER_MIN", "Мин. номер скана", int),
                ("SCAN_NUMBER_MAX", "Макс. номер скана", int),
                ("CONFIDENCE_THRESHOLD", "Порог уверенности цифры", float),
            ]),
            ("Стрелка", [
                ("ARROW_HSV_LOWER", "HSV-низ стрелки (через запятую)", list),
                ("ARROW_HSV_UPPER", "HSV-верх стрелки (через запятую)", list),
                ("ARROW_SYMMETRY_EPSILON", "Эпсилон симметрии стрелки", float),
                ("ARROW_ROI_PERCENT", "ROI стрелки (% от размера)", float),
            ]),
            ("Номер", [
                ("NUMBER_BIN_THRESH", "Порог бинаризации номера", int),
                ("NUMBER_ROI_PERCENT", "ROI номера (% от размера)", float),
            ]),
            ("Морфология", [
                ("MORPH_KERNEL_MAX_SIZE", "Макс. размер ядра", int),
                ("MORPH_DILATE_ITER", "Итераций дилатации", int),
                ("MORPH_ERODE_KERNEL_DIV_W", "Делитель ядра эрозии (ширина)", float),
                ("MORPH_ERODE_KERNEL_DIV_H", "Делитель ядра эрозии (высота)", float),
                ("MORPH_ERODE_EXTRA_ITERATIONS", "Доп. итераций эрозии", int),
            ]),
            ("HSV фильтр", [
                ("CONTOUR_HSV_LOWER", "HSV-низ контура (через запятую)", list),
                ("CONTOUR_HSV_UPPER", "HSV-верх контура (через запятую)", list),
            ]),
            ("3D/Контуры", [
                ("CONTOUR_APPROX_RATE", "Коэф. аппроксимации", float),
                ("VOLUME_DIVIDER", "Делитель объёма (мм³ в мл)", float),
                ("EASING_STRENGTH", "Сила сглаживания (0..1)", float),
            ]),
            ("Прочее", [
                ("SATURATION_THRESHOLD", "Порог насыщенности", int),
                ("TEMPLATES_DIR", "Папка шаблонов", str),
            ])
        ]
        for title, fields in tabs:
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            self.add_group(layout, title, fields)
            tab_widget.addTab(tab, title)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(tab_widget)
        btn_layout = QtWidgets.QHBoxLayout()
        buttons = [
            ("Сохранить", self.save_settings),
            ("Загрузить", self.load_settings),
            ("Сбросить", self.reset_settings),
            ("OK", self.accept)
        ]
        for text, conn in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(conn)
            btn_layout.addWidget(btn)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
        self.load_settings()

    def add_group(self, parent_layout, title, fields):
        group = QtWidgets.QGroupBox(title)
        vbox = QtWidgets.QVBoxLayout(group)
        for key, label, typ in fields:
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(QtWidgets.QLabel(label))
            if typ == bool:
                inp = QtWidgets.QCheckBox()
                inp.setChecked(getattr(Settings, key, False))
            else:
                inp = QtWidgets.QLineEdit()
                val = getattr(Settings, key, "")
                if typ == list:
                    inp.setText(",".join(map(str, val)))
                else:
                    inp.setText(str(val))
            hbox.addWidget(inp)
            vbox.addLayout(hbox)
            self.inputs[key] = (inp, typ)
        parent_layout.addWidget(group)

    def save_settings(self):
        for key, (inp, typ) in self.inputs.items():
            if typ == bool:
                val = inp.isChecked()
            else:
                txt = inp.text()
                if typ == int:
                    val = int(txt)
                elif typ == float:
                    val = float(txt)
                elif typ == list:
                    val = [int(x) if x.strip().isdigit() else float(x) for x in txt.split(",") if x.strip()]
                else:
                    val = txt
            setattr(Settings, key, val)
        Settings.save()

    def load_settings(self):
        Settings.load()
        for key, (inp, typ) in self.inputs.items():
            val = getattr(Settings, key, "")
            if typ == bool:
                inp.setChecked(val)
            elif typ == list:
                inp.setText(",".join(map(str, val)))
            else:
                inp.setText(str(val))

    def reset_settings(self):
        mod = sys.modules[Settings.__module__]
        importlib.reload(mod)
        self.load_settings()


@dataclass
class ModelSettings:
    real_width: float = Settings.DEFAULT_REAL_WIDTH
    real_height: float = Settings.DEFAULT_REAL_HEIGHT
    image_width: int = 0
    image_height: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    resample_points: int = 100


class DataReader:
    def __init__(self, directory, templates_dir=Settings.TEMPLATES_DIR):
        self.directory = Path(directory)
        self.templates_dir = Path(templates_dir)
        self.digit_templates = self._load_digit_templates()
        self.image_files = []

    def _load_digit_templates(self):
        templates = {}
        for digit in range(10):
            path = self.templates_dir / f"{digit}.png"
            template_bgr = self._imread_unicode(path)
            if template_bgr is None:
                continue
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            bbox = self._find_number_bbox(template_gray)
            norm = self._extract_and_normalize_number(template_gray, bbox)
            if norm is not None:
                templates[digit] = norm
        if not templates:
            show_error("Не удалось загрузить ни один шаблон цифры")
        else:
            logging.info(f"Загружено {len(templates)} шаблонов цифр: {sorted(templates.keys())}")
        return templates

    def _imread_unicode(self, path):
        try:
            with open(path, "rb") as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Не удалось декодировать изображение: {path}")
            return img
        except Exception as e:
            show_error(f"Ошибка чтения изображения {path}: {str(e)}", exc=e, tb=traceback.format_exc())
            return None

    def _find_number_bbox(self, gray_roi):
        try:
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            valid = [c for c in contours if cv2.contourArea(c) > Settings.MIN_CONTOUR_AREA]
            if not valid:
                show_error("Валидные контуры числа не найдены", level="warning")
                return None
            all_points = np.vstack(valid).squeeze()
            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)
            return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        except Exception as e:
            show_error(f"Ошибка поиска контура числа: {str(e)}", exc=e, tb=traceback.format_exc())
            return None

    def _extract_and_normalize_number(self, gray_roi, bbox):
        try:
            if bbox is None:
                return None
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                show_error(f"Некорректные размеры bbox: w={w}, h={h}")
                return None
            number_img = gray_roi[y:y + h, x:x + w]
            if number_img.size == 0:
                show_error("Пустое изображение числа после вырезки bbox")
                return None
            return cv2.resize(number_img, Settings.TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
        except Exception as e:
            show_error(f"Ошибка нормализации числа: {str(e)}", exc=e, tb=traceback.format_exc())
            return None

    def _find_arrow_roi(self, img_bgr):
        h, w = img_bgr.shape[:2]
        roi_h = int(h * Settings.ARROW_ROI_PERCENT)
        roi_w = int(w * Settings.ARROW_ROI_PERCENT)
        return img_bgr[0:roi_h, w - roi_w:w]

    def _extract_arrow_angle(self, roi_bgr):
        try:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            lower = np.array(Settings.ARROW_HSV_LOWER, dtype=np.uint8)
            upper = np.array(Settings.ARROW_HSV_UPPER, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                show_error("Стрелка не найдена в ROI (контуры отсутствуют)", level="warning")
                return None, None, None, None, None
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < Settings.ARROW_MIN_CONTOUR_POINTS or cv2.contourArea(contour) < Settings.ARROW_MIN_CONTOUR_AREA:
                show_error("Контур стрелки слишком мал или является шумом", level="warning")
                return None, None, None, None, None
            contour_points = contour[:, 0, :]
            M = cv2.moments(contour)
            if M['m00'] == 0:
                return None, None, None, None, None
            centroid = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            main_axis_direction = np.array([vx, vy]).reshape(-1)
            main_axis_point = np.array([x0, y0]).reshape(-1)
            normal_vector = np.array([-main_axis_direction[1], main_axis_direction[0]])
            projections_on_normal = (contour_points - main_axis_point) @ normal_vector
            widest_point_pos_idx = np.argmax(projections_on_normal)
            widest_point_neg_idx = np.argmin(projections_on_normal)
            p_wide1 = contour_points[widest_point_pos_idx]
            p_wide2 = contour_points[widest_point_neg_idx]
            barbs_midpoint = (p_wide1 + p_wide2) / 2.0
            orientation_vector = barbs_midpoint - centroid
            orientation_vector_norm = np.linalg.norm(orientation_vector)
            if orientation_vector_norm < Settings.ARROW_SYMMETRY_EPSILON:
                projections_on_main_axis = (contour_points - centroid) @ main_axis_direction
                tip_candidate_idx = np.argmax(projections_on_main_axis)
                base_candidate_idx = np.argmin(projections_on_main_axis)
                tip_candidate = contour_points[tip_candidate_idx]
                base_candidate = contour_points[base_candidate_idx]
                dist_tip_to_barbs = np.linalg.norm(tip_candidate - barbs_midpoint)
                dist_base_to_barbs = np.linalg.norm(base_candidate - barbs_midpoint)
                arrow_direction = main_axis_direction if dist_tip_to_barbs < dist_base_to_barbs else -main_axis_direction
            else:
                arrow_direction = orientation_vector / orientation_vector_norm
            final_projections = (contour_points - centroid) @ arrow_direction
            tip_idx = np.argmax(final_projections)
            base_idx = np.argmin(final_projections)
            tip = contour_points[tip_idx]
            base = contour_points[base_idx]
            vec = tip - base
            if np.linalg.norm(vec) == 0:
                return None, None, None, None, None
            angle_rad = np.arctan2(vec[0], -vec[1])
            angle_deg = np.degrees(angle_rad)
            angle_deg = (angle_deg + 360) % 360
            return angle_deg, base, tip, vec, mask
        except Exception as e:
            show_error(f"Критическая ошибка извлечения угла стрелки: {str(e)}", exc=e, tb=traceback.format_exc())
            return None, None, None, None, None

    def _find_number_roi(self, img_bgr):
        h, w = img_bgr.shape[:2]
        roi_h = int(h * Settings.NUMBER_ROI_PERCENT)
        roi_w = int(w * Settings.NUMBER_ROI_PERCENT)
        roi_bgr = img_bgr[0:roi_h, 0:roi_w]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return roi_gray

    def _extract_digits_from_roi(self, roi_gray):
        _, thresh = cv2.threshold(roi_gray, Settings.NUMBER_BIN_THRESH, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > Settings.MIN_CONTOUR_AREA]
        bboxes = sorted(bboxes, key=lambda b: b[0])
        return [cv2.resize(roi_gray[y:y+h, x:x+w], Settings.TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA) for x, y, w, h in bboxes]

    def _recognize_digit(self, digit_img):
        if not self.digit_templates:
            show_error("Нет шаблонов для распознавания", level="warning")
            return None, None
        best_digit, best_val = None, float('-inf')
        for digit, template in self.digit_templates.items():
            if digit_img.shape != template.shape:
                continue
            res = cv2.matchTemplate(digit_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_digit = digit
        if best_val < Settings.CONFIDENCE_THRESHOLD:
            show_error(f"Цифра не распознана, уверенность: {best_val:.2f}", level="warning")
            return None, best_val
        return best_digit, best_val

    def _extract_number(self, roi_gray):
        try:
            digit_imgs = self._extract_digits_from_roi(roi_gray)
            digits = [self._recognize_digit(img)[0] for img in digit_imgs if self._recognize_digit(img)[0] is not None]
            if not digits:
                return None
            if len(digits) == 1:
                return digits[0]
            elif len(digits) == 2:
                return digits[0] * 10 + digits[1]
            else:
                show_error(f"Распознано {len(digits)} цифр, ожидалось 1 или 2. Используется только первая цифра.", level="warning")
                return digits[0]
        except Exception as e:
            show_error(f"Ошибка извлечения номера: {str(e)}", exc=e, tb=traceback.format_exc())
            return None

    def read_images(self):
        try:
            all_image_paths = sorted([p for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"] for p in self.directory.glob(ext)], key=lambda p: p.name)
            image_data = []
            image_shape = None
            for file_path in all_image_paths:
                img_bgr = self._imread_unicode(file_path)
                if img_bgr is None:
                    continue
                if image_shape is None:
                    image_shape = img_bgr.shape
                elif img_bgr.shape != image_shape:
                    raise ValueError(f"Обнаружено изображение с другим разрешением: {file_path.name}")
                roi_arrow = self._find_arrow_roi(img_bgr)
                angle = self._extract_arrow_angle(roi_arrow)[0]
                roi_num = self._find_number_roi(img_bgr)
                number = self._extract_number(roi_num)
                if angle is None and number is None:
                    show_error(f"Не удалось определить ни угол стрелки, ни номер скана в файле: {file_path.name}", level="warning")
                    continue
                image_data.append({'img': img_bgr, 'angle': angle, 'number': number, 'file': file_path})
            numbers = [d['number'] for d in image_data]
            are_numbers_valid = (
                all(n is not None for n in numbers) and
                all(Settings.SCAN_NUMBER_MIN <= n <= Settings.SCAN_NUMBER_MAX for n in numbers) and
                len(set(numbers)) == len(numbers)
            )
            sorted_data = []
            if are_numbers_valid:
                sorted_data = sorted(image_data, key=lambda d: d['number'])
                logging.info(f"Данные успешно отсортированы по номерам сканов: { [d['number'] for d in sorted_data] }.")
            else:
                show_error(
                    "Не удалось отсортировать данные по номерам сканов (отсутствуют, не уникальны или вне диапазона). "
                    "Попытка сортировки по углам стрелок...",
                    level="warning"
                )
                angles = [d['angle'] for d in image_data]
                are_angles_valid = all(a is not None for a in angles) and len(set(angles)) == len(angles)
                if are_angles_valid:
                    sorted_data = sorted(image_data, key=lambda d: d['angle'])
                    logging.info("Данные отсортированы по углам стрелок (запасной метод).")
                else:
                    raise ValueError("Не удалось однозначно определить порядок сканов: и номера, и углы некорректны или не уникальны.")
            if not sorted_data:
                raise ValueError("Не найдено ни одного валидного изображения для сортировки.")
            self.image_files = [d['file'] for d in sorted_data]
            return [d['img'] for d in sorted_data], [d['angle'] for d in sorted_data], [d['number'] for d in sorted_data], image_shape
        except Exception as e:
            show_error(f"Ошибка чтения данных: {str(e)}", exc=e, tb=traceback.format_exc())
            raise


class ImageProcessor:
    def __init__(self, saturation_threshold=Settings.SATURATION_THRESHOLD):
        self.saturation_threshold = saturation_threshold

    def _create_hsv_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Создает маску на основе HSV диапазона из Settings."""
        lower = np.array(Settings.CONTOUR_HSV_LOWER, dtype=np.uint8)
        upper = np.array(Settings.CONTOUR_HSV_UPPER, dtype=np.uint8)
        return cv2.inRange(hsv_image, lower, upper)

    def _apply_morphology(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Применяет последовательность морфологических операций из Settings."""
        h, w = image_shape[:2]
        kernel_size = min(Settings.MORPH_KERNEL_MAX_SIZE, h, w)
        if kernel_size < 1:
            show_error("Размер ядра морфологии слишком мал", level="warning")
            return mask
        kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel_circle, iterations=Settings.MORPH_DILATE_ITER)
        erode_kernel_w = max(1, int(kernel_size / Settings.MORPH_ERODE_KERNEL_DIV_W))
        erode_kernel_h = max(1, int(kernel_size / Settings.MORPH_ERODE_KERNEL_DIV_H))
        kernel_circle_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_w, erode_kernel_h))
        mask = cv2.erode(
            mask, 
            kernel_circle_small, 
            iterations=Settings.MORPH_DILATE_ITER + Settings.MORPH_ERODE_EXTRA_ITERATIONS
        )
        return mask

    def _find_largest_contour(self, mask: np.ndarray) -> np.ndarray:
        """Находит самый большой внешний контур."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _validate_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """Проверяет контур на минимальное количество точек и площадь."""
        h, w = image_shape[:2]
        if len(contour) < Settings.MIN_CONTOUR_POINTS:
            logging.warning("Контур слишком мал (недостаточно точек)")
            return False
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        bbox_area = bbox_w * bbox_h
        contour_area = cv2.contourArea(contour)
        is_valid_area = False
        if bbox_area > 0 and contour_area / bbox_area >= Settings.MIN_CONTOUR_AREA_PERCENTAGE:
            is_valid_area = True
        else:
            is_near_border = False
            # contour.squeeze() преобразует (N, 1, 2) в (N, 2)
            for point in contour.squeeze(): 
                px, py = point[0], point[1]
                if (px <= Settings.BORDER_TOLERANCE_PX or 
                    px >= w - 1 - Settings.BORDER_TOLERANCE_PX or 
                    py <= Settings.BORDER_TOLERANCE_PX or 
                    py >= h - 1 - Settings.BORDER_TOLERANCE_PX):
                    is_near_border = True
                    break
            if is_near_border:
                logging.warning("Похоже, один из контуров выходит за границу изображения. Контур будет интерполирован по его ближайшим соседям")
            else:
                logging.warning("Контур слишком мал (не соответствует проценту площади) и не у границы. Контур будет интерполирован по его ближайшим соседям")
            return False 
        return is_valid_area

    def _approximate_and_resample(self, contour: np.ndarray, approximation_rate: float, n_points: int) -> np.ndarray:
        """Аппроксимирует и ресэмплирует контур."""
        arclen = cv2.arcLength(contour, True)
        epsilon = arclen * approximation_rate
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return resample_contour(approx, n_points=n_points)

    def process_image(self, img: np.ndarray, approximation_rate: float = Settings.CONTOUR_APPROX_RATE) -> np.ndarray:
        """
        Обрабатывает изображение для извлечения контура.
        Args:
            img: Входное BGR изображение.
            approximation_rate: Коэффициент аппроксимации.
        Returns:
            Обработанный контour (np.ndarray) или None.
        """
        try:
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = self._create_hsv_mask(hsv)
            mask = self._apply_morphology(mask, (h, w))
            mask = cv2.ximgproc.thinning(mask)
            contour = self._find_largest_contour(mask)
            if contour is None:
                logging.warning("Контур не найден")
                return None
            if not self._validate_contour(contour, (h, w)):
                return None
            return self._approximate_and_resample(contour, approximation_rate, n_points=150)
        except Exception as e:
            show_error(f"Ошибка обработки изображения: {str(e)}", exc=e, tb=traceback.format_exc())
            return None


class ModelBuilder:
    def __init__(self, image_width, image_height, real_width=Settings.DEFAULT_REAL_WIDTH, real_height=Settings.DEFAULT_REAL_HEIGHT, n_resample_points=150):
        self.settings = ModelSettings(
            real_width=real_width,
            real_height=real_height,
            image_width=image_width,
            image_height=image_height,
            scale_x=image_width / real_width if real_width else 1.0,
            scale_y=image_height / real_height if real_height else 1.0,
            resample_points=n_resample_points
        )
        self.points = None
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.REAL_WIDTH = real_width
        self.REAL_HEIGHT = real_height
        self.scale_x = self.settings.scale_x
        self.scale_y = self.settings.scale_y
        self.n_resample_points = n_resample_points
        logging.info(
            f"Default scales calculated: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm"
        )

    def estimate_resample_points_from_contours(self, contours, angles=None, center=None, min_angle_deg: float = None) -> int:
        try:
            if not contours:
                return self.n_resample_points
            if min_angle_deg is None:
                min_angle_deg = Settings.MIN_ANGLE_BETWEEN_CONTOURS
            dtheta = max(min_angle_deg, 1e-6) * np.pi / 180.0
            if center is None:
                center_x = self.IMAGE_WIDTH // 2
                center_y = self.IMAGE_HEIGHT // 2
            else:
                center_x, center_y = center
            radii_mm = []
            perims_mm = []
            for c in contours:
                if c is None:
                    continue
                pts = np.array(c).reshape(-1, 2)
                if len(pts) < 3:
                    continue
                rx_pix = np.median(np.abs(pts[:, 0] - center_x))
                r_mm = rx_pix / max(self.scale_x, 1e-6)
                if r_mm > 0:
                    radii_mm.append(r_mm)
                dx_pix = np.diff(np.r_[pts[:, 0], pts[0, 0]])
                dy_pix = np.diff(np.r_[pts[:, 1], pts[0, 1]])
                dx_mm = dx_pix / max(self.scale_x, 1e-6)
                dy_mm = dy_pix / max(self.scale_y, 1e-6)
                perim_mm = float(np.sum(np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)))
                if perim_mm > 0:
                    perims_mm.append(perim_mm)
            if not perims_mm:
                return self.n_resample_points
            radius_mm = float(np.median(radii_mm)) if radii_mm else max(perims_mm) / (2 * np.pi)
            arc_step_mm = max(radius_mm * dtheta, 1e-3)
            perim_med_mm = float(np.median(perims_mm))
            est_points = int(np.clip(round(perim_med_mm * 0.5 / arc_step_mm), 50, 300))
            if est_points < 12:
                est_points = 12
            if est_points % 2 == 1:
                est_points += 1
            old = self.n_resample_points
            self.n_resample_points = est_points
            logging.info(
                f"Adaptive resample points: radius≈{radius_mm:.2f}mm, Δθ={min_angle_deg:.3f}°, arc_step≈{arc_step_mm:.3f}mm, "
                f"perimeter≈{perim_med_mm:.1f}mm → n_points={est_points} (from {old})"
            )
            return est_points
        except Exception as e:
            logging.warning(f"Не удалось оценить адаптивное число точек: {e}")
            return self.n_resample_points

    def prepare_contours(self, contours, scan_numbers, angles: List[float], center=None):
        if not contours or not scan_numbers:
            raise ValueError("Нет валидных контуров или номеров сканов")
        
        try:
            self.estimate_resample_points_from_contours(
                contours,
                angles=angles,
                center=center,
                min_angle_deg=Settings.MIN_ANGLE_BETWEEN_CONTOURS,
            )
        except Exception:
            pass
        
        # Подготовка начальных контуров
        initial_contours = [
            resample_contour(c, self.n_resample_points) if c is not None else None
            for c in contours
        ]
        original_count = len(initial_contours)
        
        # Заполнение пропущенных контуров
        filled_contours, filled_angles, filled_scans = self._fill_missing_contours(
            initial_contours, list(angles), list(scan_numbers)
        )
        
        # Добавление контуров для уменьшения угла между ними
        final_contours, final_angles, final_scans, is_original = self._add_missing_angles(
            filled_contours, filled_angles, filled_scans, center
        )
        
        # Сохранение результатов
        self.is_original_contour = is_original
        self.final_contours = final_contours
        self.final_angles = final_angles
        self.final_scan_numbers = final_scans
        
        logging.info(f"Итеративная интерполяция: {original_count} исходных контуров -> {len(final_contours)} контуров")
        return final_contours, final_angles, final_scans, is_original

    def _fill_missing_contours(self, contours, angles, scan_numbers):
        n = len(contours)
        if n == 0:
            raise ValueError("Список контуров пуст")
        
        has_any_valid = any(c is not None for c in contours)
        if not has_any_valid:
            raise ValueError("Нет ни одного валидного контура для интерполяции")
        
        # Создаем копии для модификации
        contours_list = list(contours)
        angles_list = list(angles)
        scans_list = list(scan_numbers)
        
        # Ищем и заполняем пропуски
        idx = 0
        while idx < n:
            if contours_list[idx] is not None:
                idx += 1
                continue
                
            # Находим предыдущий и следующий валидные контуры
            prev_idx, next_idx = self._find_valid_neighbors(contours_list, idx)
            
            # Если не удалось найти валидные контуры с обеих сторон
            if prev_idx == idx or next_idx == idx:
                valid_idx = prev_idx if contours_list[prev_idx] is not None else next_idx
                contours_list[idx] = resample_contour(contours_list[valid_idx], self.n_resample_points)
                scans_list[idx] = -1
                idx += 1
                continue
                
            # Интерполируем между найденными контурами
            self._interpolate_gap(contours_list, angles_list, scans_list, idx, prev_idx, next_idx)
            idx = next_idx + 1 if next_idx >= idx else idx + 1
        
        return contours_list, angles_list, scans_list

    def _find_valid_neighbors(self, contours, current_idx):
        n = len(contours)
        prev_idx = (current_idx - 1) % n
        while contours[prev_idx] is None and prev_idx != current_idx:
            prev_idx = (prev_idx - 1) % n
        
        next_idx = (current_idx + 1) % n
        while contours[next_idx] is None and next_idx != current_idx:
            next_idx = (next_idx + 1) % n
        
        return prev_idx, next_idx

    def _interpolate_gap(self, contours, angles, scans, current_idx, prev_idx, next_idx):
        gap_indices = []
        j = (prev_idx + 1) % len(contours)
        while j != next_idx:
            gap_indices.append(j)
            j = (j + 1) % len(contours)
        
        if not gap_indices:
            return
            
        c_prev = contours[prev_idx]
        c_next = contours[next_idx]
        c_next_for_interp = c_next
        
        for pos, gi in enumerate(gap_indices, start=1):
            alpha = pos / (len(gap_indices) + 1)
            try:
                interp = self.interpolate_contour(c_prev, c_next_for_interp, alpha, linear_alpha=alpha)
            except Exception:
                interp = resample_contour(c_prev, self.n_resample_points)
            contours[gi] = interp
            scans[gi] = -1

    def _add_missing_angles(self, contours, angles, scans, center):
        n = len(contours)
        is_original = [sn != -1 for sn in scans]
        added = True
        
        while added:
            added = False
            new_contours = []
            new_angles = []
            new_scans = []
            new_is_original = []
            
            for i in range(n):
                c1 = contours[i]
                c2 = contours[(i + 1) % n]
                a1 = angles[i]
                a2 = angles[(i + 1) % n]
                
                # Добавляем текущий контур
                new_contours.append(c1)
                new_angles.append(a1)
                new_scans.append(scans[i])
                new_is_original.append(is_original[i])
                
                # Проверяем необходимость добавления промежуточных контуров
                angle_diff = a2 - a1
                is_wrap = angle_diff < 0
                if is_wrap:
                    angle_diff += 180.0
                    
                if angle_diff > Settings.MIN_ANGLE_BETWEEN_CONTOURS:
                    c2_for_interp = c2
                    if is_wrap:
                        center_x = center[0] if center else self.IMAGE_WIDTH // 2
                        c2_for_interp = c2.copy()
                        c2_for_interp[..., 0] = 2 * center_x - c2_for_interp[..., 0]
                    
                    n_to_insert = int(angle_diff // Settings.MIN_ANGLE_BETWEEN_CONTOURS)
                    n_segments = n_to_insert + 1
                    
                    for j in range(1, n_to_insert + 1):
                        linear_alpha = j / n_segments
                        f_ease = -2 * (linear_alpha**3) + 3 * (linear_alpha**2)
                        k = Settings.EASING_STRENGTH
                        blended_alpha = (1 - k) * linear_alpha + k * f_ease
                        interp_angle = a1 + linear_alpha * angle_diff
                        if interp_angle >= 180.0:
                            interp_angle -= 180.0
                        
                        interp_contour = self.interpolate_contour(c1, c2_for_interp, blended_alpha, linear_alpha=linear_alpha)
                        new_contours.append(interp_contour)
                        new_angles.append(interp_angle)
                        new_scans.append(-1)
                        new_is_original.append(False)
                    
                    added = True
            
            if not added:
                break
                
            # Сортируем по углам
            zipped = list(zip(new_angles, new_contours, new_is_original, new_scans))
            zipped.sort(key=lambda x: x[0])
            angles, contours, is_original, scans = zip(*zipped)
            angles = list(angles)
            contours = list(contours)
            is_original = list(is_original)
            scans = list(scans)
            n = len(contours)
        
        return contours, angles, scans, is_original

    def build_point_cloud_and_mesh(self, contours, angles, center=None):
        if center is None:
            center = (
                self.IMAGE_WIDTH // 2,
                self.IMAGE_HEIGHT // 2
            )
        contours_as_3d_points = []
        for i, contour in enumerate(contours):
            current_contour_3d_points = [
                [
                    (point[0][0] - center[0]) / self.scale_x,
                    (center[1] - point[0][1]) / self.scale_y,
                    0.0
                ]
                for point in contour
            ]
            contours_as_3d_points.append(np.array(current_contour_3d_points))
        points_list = []
        for i, contour_3d_points_array in enumerate(contours_as_3d_points):
            angle_rad = angles[i] * np.pi / 180
            for p in contour_3d_points_array:
                x_3d = p[0] * np.cos(angle_rad)
                y_3d = p[1]
                z_3d = p[0] * np.sin(angle_rad)
                points_list.append([x_3d, y_3d, z_3d])
        points = np.array(points_list)
        if points.shape[0] < 4:
            raise ValueError(f"Недостаточно точек для триангуляции: {points.shape[0]}")
        self.points = points
        self.individual_contour_3d_points = contours_as_3d_points
        self.angles = angles
        cloud = pv.PolyData(points)
        grid = cloud.delaunay_3d(alpha=0.01)
        surf = grid.extract_geometry()
        logging.info(
            f"Full mesh: {surf.n_faces} faces"
        )
        return surf

    def build_model(self, contours, scan_numbers, angles=None, center=None):
        contours_list, angles_list, scans_list, is_original = self.prepare_contours(contours, scan_numbers, angles, center)
        return self.build_point_cloud_and_mesh(contours_list, angles_list, center)

    def interpolate_contour(self, contour1, contour2, alpha, linear_alpha=None):
        """
        Интерполирует между двумя контурами с весом alpha (0-1).
        Дополнительно корректирует размер интерполированной формы для предотвращения
        "усыхания" в промежуточных кадрах.
        """
        if linear_alpha is None:
            linear_alpha = alpha
        aligned_c1, aligned_c2 = self._get_aligned_contours(contour1, contour2)
        interpolated_raw = (1 - alpha) * aligned_c1 + alpha * aligned_c2
        x1, y1, w1, h1 = cv2.boundingRect(aligned_c1.astype(np.float32))
        x2, y2, w2, h2 = cv2.boundingRect(aligned_c2.astype(np.float32))
        target_w = (1 - linear_alpha) * w1 + linear_alpha * w2
        target_h = (1 - linear_alpha) * h1 + linear_alpha * h2
        if interpolated_raw.shape[0] < 3:
            return interpolated_raw.astype(np.int32).reshape(-1, 1, 2)
        x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(interpolated_raw.astype(np.float32))
        interpolated = interpolated_raw
        if w_curr > 1 and h_curr > 1:
            scale_x = target_w / w_curr
            scale_y = target_h / h_curr
            center_curr = np.mean(interpolated_raw, axis=0)
            centered_interpolated = interpolated_raw - center_curr
            scaled_interpolated = centered_interpolated * np.array([scale_x, scale_y])
            interpolated = scaled_interpolated + center_curr
        return interpolated.astype(np.float32).reshape(-1, 1, 2)

    def _get_aligned_contours(self, contour1, contour2):
        n_points = self.n_resample_points
        c1 = resample_contour(contour1, n_points).squeeze()
        c2 = resample_contour(contour2, n_points).squeeze()
        cm1 = np.mean(c1, axis=0)
        cm2 = np.mean(c2, axis=0)
        c1_centered = c1 - cm1
        c2_centered = c2 - cm2
        best_dist = np.inf
        best_shift = 0
        best_reversed = False
        for is_reversed in [False, True]:
            c2_orient = c2_centered[::-1] if is_reversed else c2_centered
            for shift in range(n_points):
                c2_shifted = np.roll(c2_orient, -shift, axis=0)
                dist = np.sum((c1_centered - c2_shifted) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_shift = shift
                    best_reversed = is_reversed
        if best_reversed:
            c2_reordered = c2[::-1]
        else:
            c2_reordered = c2
        aligned_c2 = np.roll(c2_reordered, -best_shift, axis=0)
        return c1, aligned_c2

    @staticmethod
    def _mid_profile(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = min(len(a), len(b))
        return 0.5 * (a[:n] + b[:n])

    @staticmethod
    def _first_moment_of_area(contour: np.ndarray) -> float:
        if contour is None or len(contour) < 3:
            return 0.0
        points = contour.reshape(-1, 2)
        rolled_points = np.roll(points, -1, axis=0)
        xi, yi = points[:, 0], points[:, 1]
        x_next, y_next = rolled_points[:, 0], rolled_points[:, 1]
        cross_product_term = xi * y_next - x_next * yi
        x_sum_term = xi + x_next
        moment = np.sum(x_sum_term * cross_product_term) / 6.0
        return abs(moment)

    def volume_radial_integration(self, contours: List[np.ndarray], angles: List[float], center: Tuple[float, float] = None) -> float:
        if len(contours) < 2:
            return 0.0
        if center is None:
            center_x = self.IMAGE_WIDTH / 2.0
            center_y = self.IMAGE_HEIGHT / 2.0
        else:
            center_x = center[0]
            center_y = center[1]
        sorted_data = sorted(zip(angles, contours), key=lambda x: x[0])
        sorted_angles, sorted_contours = zip(*sorted_data)
        moments_mm3 = []
        for contour_px in sorted_contours:
            if contour_px is None or len(contour_px) < 3:
                moments_mm3.append(0.0)
                continue
            points_px = contour_px.reshape(-1, 2).astype(np.float32)
            points_mm = np.zeros_like(points_px)
            points_mm[:, 0] = (points_px[:, 0] - center_x) / self.scale_x
            points_mm[:, 1] = (points_px[:, 1] - center_y) / self.scale_y
            moment = self._first_moment_of_area(points_mm)
            moments_mm3.append(moment)
        angles_rad = np.deg2rad(sorted_angles)
        volume = float(np.trapz(y=moments_mm3, x=angles_rad))
        return abs(volume)


class DebugViewer(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Отладочный просмотрщик изображений")
        self.setGeometry(200, 200, 1000, 700)
        self.images = self.angles = self.scan_numbers = self.contours = self.image_files = []
        self.colors = []
        self.current_index = 0
        self.show_interpolated = True
        self.filtered_indices = []
        self.scan_to_image_map: Dict[int, int] = {} 
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        info_layout = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel("Нет данных")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        info_layout.addWidget(self.info_label)
        self.interp_checkbox = QtWidgets.QCheckBox("Показывать интерполированные")
        self.interp_checkbox.setChecked(True)
        self.interp_checkbox.stateChanged.connect(self.on_interp_checkbox_changed)
        info_layout.addWidget(self.interp_checkbox)
        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("← Предыдущее")
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)
        self.next_button = QtWidgets.QPushButton("Следующее →")
        self.next_button.clicked.connect(self.show_next)
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
        self.details_label = QtWidgets.QLabel("Детали распознавания:")
        self.details_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.details_label)
        help_text = ("Подсказки: ←/→ навигация, колесико мыши - масштаб, 0 - сброс масштаба, "
                     "Esc - закрыть, перетаскивание мышью - перемещение")
        help_label = QtWidgets.QLabel(help_text)
        help_label.setStyleSheet("font-size: 10px; color: gray; font-style: italic;")
        layout.addWidget(help_label)

    def set_data(self, images: List[np.ndarray], scan_numbers: List[int], contours: List[np.ndarray], angles: List[float], colors: List[Tuple[int, int, int]], scan_to_image_map: Dict[int, int]):
        self.images = images
        self.scan_numbers = scan_numbers
        self.contours = contours
        self.angles = angles
        self.colors = colors
        self.scan_to_image_map = scan_to_image_map
        self.current_index = 0
        self.update_filtered_indices()
        if self.filtered_indices:
            self.show_current_image()

    def update_filtered_indices(self):
        if self.show_interpolated:
            self.filtered_indices = list(range(len(self.contours)))
        else:
            self.filtered_indices = [i for i, sn in enumerate(self.scan_numbers) if sn != -1]
        if self.filtered_indices and self.current_index >= len(self.filtered_indices):
            self.current_index = 0
        self.prev_button.setEnabled(bool(self.filtered_indices))
        self.next_button.setEnabled(bool(self.filtered_indices))

    def on_interp_checkbox_changed(self, state):
        self.show_interpolated = bool(state)
        self.update_filtered_indices()
        self.show_current_image()

    def show_current_image(self):
        if not self.filtered_indices:
            self.info_label.setText("Нет данных")
            self.scene.clear()
            self.details_label.setText("")
            return
        idx = self.filtered_indices[self.current_index]
        contour = self.contours[idx]
        angle = self.angles[idx] if idx < len(self.angles) else "N/A"
        color = self.colors[idx] if self.colors and idx < len(self.colors) else (200, 200, 200)
        scan_number = self.scan_numbers[idx]
        is_original = scan_number != -1
        if is_original:
            image_index = self.scan_to_image_map.get(scan_number)
            if image_index is not None and 0 <= image_index < len(self.images):
                img = self.images[image_index].copy()
            else:
                logging.warning(f"Не удалось найти оригинальное изображение для скана №{scan_number} (индекс: {image_index}).")
                if self.images:
                    img = np.zeros_like(self.images[0], dtype=np.uint8)
                else:
                    img = np.zeros((100, 100, 3), dtype=np.uint8)
            info_text = f"Изображение {self.current_index + 1}/{len(self.filtered_indices)} (Оригинал) | Скан: {scan_number} | Угол: {angle:.2f}°"
        else:
            if self.images:
                img = np.zeros_like(self.images[0], dtype=np.uint8)
            else:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            info_text = f"Изображение {self.current_index + 1}/{len(self.filtered_indices)} (Интерполированный) | Угол: {angle:.2f}°"
        if contour is not None and len(contour) > 0:
            cv2.drawContours(img, [contour.astype(np.int32)], -1, color, 2)
        h, w = img.shape[:2]
        display_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(display_img_rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.info_label.setText(info_text)
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
        if is_original:
            details.append(f"Распознан номер: {scan_number}")
        else:
            details.append("Номер не распознан (интерполированный)")
        details.append(f"Распознан угол: {angle:.2f}°" if isinstance(angle, (int, float)) else "Угол не распознан")
        details.append(f"Размер: {w}x{h} пикселей")
        self.details_label.setText(" | ".join(details))

    def show_previous(self):
        if not self.filtered_indices:
            return
        self.current_index = (self.current_index - 1) % len(self.filtered_indices)
        self.show_current_image()

    def show_next(self):
        if not self.filtered_indices:
            return
        self.current_index = (self.current_index + 1) % len(self.filtered_indices)
        self.show_current_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene and not self.scene.sceneRect().isEmpty():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.show_previous()
        elif key == Qt.Key.Key_Right:
            self.show_next()
        elif key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_0:
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.view.scale(factor, factor)


def rgb_to_bgr(color):
    return (color[2], color[1], color[0])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intraocular 3D Volume Calculator")
        self.setGeometry(100, 100, 800, 600)
        self.reader = DataReader(".")
        self.processor = ImageProcessor()
        self.builder = None
        self.plotter = None
        self.progress_bar = None
        self.scan_to_image_map: Dict[int, int] = {} 
        self.debug_viewer = DebugViewer(self)
        self.image_processor = ImageProcessor()
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
        open_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        debug_action = QAction("&Отладочный просмотрщик...", self)
        debug_action.setShortcut("Ctrl+D")
        debug_action.triggered.connect(self.open_debug_viewer)
        file_menu.addAction(debug_action)
        file_menu.addSeparator()
        exit_action = QAction("&Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        tools_menu = menubar.addMenu("&Инструменты")
        settings_action = QAction("&Расширенные настройки...", self)
        settings_action.setShortcut("Ctrl+Shift+S")
        settings_action.triggered.connect(self.open_settings_dialog)
        tools_menu.addAction(settings_action)
        help_menu = menubar.addMenu("&Справка")
        about_action = QAction("&О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "О программе", 
                                    "3D Scan Processor\n"
                                    "Программа для обработки 3D сканов и расчета объема\n"
                                    "Использует PyVista для 3D визуализации\n"
                                    "Версия 1.0")

    def _set_progress(self, visible: bool, maximum: int = 100, value: int = 0, text: str = ""):
        self.progress_bar.setVisible(visible)
        if text:
            self.progress_bar.setFormat(text)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()

    def select_folder(self):
        try:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
            if not folder:
                return
            self.reader.directory = Path(folder)
            images, arrow_angles, scan_numbers, image_shape = self.reader.read_images()
            if image_shape is None:
                raise ValueError("Не удалось определить разрешение изображений")
            image_height, image_width = image_shape[:2]
            self.scan_to_image_map = {num: i for i, num in enumerate(scan_numbers) if num is not None}
            if all(n is not None for n in scan_numbers):
                N = len(images)
                if N == 0: raise ValueError("Изображения не найдены.")
                angles = [i * (180.0 / N) for i in range(N)]
                logging.info("Порядок определен по номерам сканов, используются сгенерированные углы.")
            else:
                angles = arrow_angles
                logging.info("Порядок определен по углам, используются распознанные углы.")
            self.builder = ModelBuilder(image_width, image_height, n_resample_points=150)
            self._set_progress(True, len(images), 0, "Обработка изображений: %p%")
            contours = [self.image_processor.process_image(img) for img in images]
            self._set_progress(False)
            if not contours:
                raise ValueError("Не удалось извлечь ни одного контура")
            prepared = self.builder.prepare_contours(contours, scan_numbers, angles)
            contours_list, angles_list, scans_list, is_original = prepared
            mesh = self.builder.build_point_cloud_and_mesh(contours_list, angles_list)
            half_contours = []
            half_angles = []
            for contour_px, ang in zip(contours_list, angles_list):
                if contour_px is None or len(contour_px) < 3:
                    continue
                n_full = contour_px.shape[0]
                n_half = n_full // 2 + 1
                right_half = contour_px[:n_half]
                left_half = contour_px[n_half - 1:]
                half_contours.append(right_half)
                half_angles.append(ang)
                half_contours.append(left_half)
                half_angles.append((ang + 180.0) % 360.0)
            vol_radial = self.builder.volume_radial_integration(half_contours, half_angles)
            volume_mm3 = float(vol_radial)
            volume_ml = volume_mm3 / Settings.VOLUME_DIVIDER
            self.last_images = images
            self.last_contours = self.builder.final_contours
            self.last_angles = self.builder.final_angles
            self.last_scan_numbers = self.builder.final_scan_numbers
            self.last_image_files = [f.name for f in self.reader.image_files]
            is_original_flags = getattr(self.builder, 'is_original_contour', [True] * len(self.last_contours))
            original_count = sum(1 for flag in is_original_flags if flag)
            original_colors_map = []
            if original_count > 0:
                original_colors_map = plt.cm.hsv(np.linspace(0, 1, original_count, endpoint=False))[:, :3]
            self.last_contour_colors_rgb = []
            self.last_contour_colors_bgr = []
            original_idx = 0
            for is_orig in is_original_flags:
                if is_orig and original_idx < len(original_colors_map):
                    color_np = original_colors_map[original_idx] * 255
                    color_rgb = tuple(map(int, color_np))
                    self.last_contour_colors_rgb.append(color_rgb)
                    self.last_contour_colors_bgr.append(rgb_to_bgr(color_rgb))
                    original_idx += 1
                else:
                    self.last_contour_colors_rgb.append((180, 180, 180))
                    self.last_contour_colors_bgr.append((180, 180, 180))
            self.visualize_model(mesh, colors=self.last_contour_colors_rgb)
            self.volume_label.setText(
                f"Объём: {volume_mm3:.4f} мм³ ({volume_ml:.5f} мл) | Radial(halves): {vol_radial:.4f}"
            )
        except Exception as e:
            show_error(f"Ошибка обработки: {str(e)}", exc=e, tb=traceback.format_exc())
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)

    def visualize_model(self, mesh, colors=None):
        try:
            if self.plotter is not None:
                self.plotter.clear()
            else:
                self.plotter = self.vtk_widget
            if hasattr(self.builder, "individual_contour_3d_points") and hasattr(self.builder, "angles"):
                groups = self.builder.individual_contour_3d_points
                angles = self.builder.angles
                n_contours = len(groups)
                use_generated_colors = colors is not None and len(colors) == n_contours
                for i, group_points_raw in enumerate(groups):
                    if i >= len(angles):
                        continue
                    angle_rad = angles[i] * np.pi / 180
                    x_3d = group_points_raw[:, 0] * np.cos(angle_rad)
                    y_3d = group_points_raw[:, 1]
                    z_3d = group_points_raw[:, 0] * np.sin(angle_rad)
                    rotated = np.stack([x_3d, y_3d, z_3d], axis=-1)
                    if len(rotated) > 1:
                        poly = pv.lines_from_points(rotated, close=True)
                        color_to_use = (200, 200, 200)
                        line_width = 2
                        opacity = 0.8
                        if use_generated_colors:
                            color_to_use = colors[i]
                            if color_to_use == (180, 180, 180):
                                line_width = 1.5
                                opacity = 0.2
                            else:
                                line_width = 3
                                opacity = 1.0
                        self.plotter.add_mesh(poly, color=color_to_use, line_width=line_width, 
                                              opacity=opacity, name=f"scanline_{i}")
            self.plotter.set_background((0.1, 0.1, 0.15))
            self.plotter.reset_camera()
            axes = pv.AxesAssembly(label_color="white", label_size=12)
            self.plotter.add_orientation_widget(axes)
            self.plotter.update()
        except Exception as e:
            show_error(f"Ошибка визуализации: {str(e)}", exc=e, tb=traceback.format_exc())

    def copy_volume(self, event):
        QtWidgets.QApplication.clipboard().setText(self.volume_label.text())

    def open_debug_viewer(self):
        if not hasattr(self, "last_images") or not self.last_images:
            show_error("Нет данных для отладки. Сначала выберите папку с изображениями.")
            return
        self.debug_viewer.set_data(
            images=self.last_images,
            scan_numbers=self.last_scan_numbers,
            contours=self.last_contours,
            angles=self.last_angles,
            colors=self.last_contour_colors_bgr,
            scan_to_image_map=self.scan_to_image_map 
        )
        self.debug_viewer.show()
        self.debug_viewer.raise_()
        self.debug_viewer.activateWindow()

    def open_settings_dialog(self):
        dlg = SettingsDialog(self)
        dlg.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(True)
    window = MainWindow()
    window.show()
    app.exec()
