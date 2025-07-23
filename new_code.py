import logging
from dataclasses import dataclass, asdict, replace, field
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

LOG_FILENAME = "scan_processor.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

log_file_handler = logging.FileHandler(LOG_FILENAME, mode="a", encoding="utf-8")
logging.basicConfig(handlers=[log_file_handler], level=LOG_LEVEL, format=LOG_FORMAT)

def show_error(message: str, level: str = "critical"):
    mapping = {"critical": (logging.error, QMessageBox.critical, "CRITICAL ERROR"),
               "warning": (logging.warning, QMessageBox.warning, "WARNING")}
    log_func, dialog_func, prefix = mapping[level]
    log_func(message)
    app = QApplication.instance()
    if app:
        dialog_func(None, "Ошибка" if level == "critical" else "Внимание", message)
    else:
        print(f"{prefix}: {message}")

def resample_contour(contour: np.ndarray, n_points: int = 120) -> np.ndarray:
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
            t0, t1 = cumulative[idx - 1], cumulative[idx]
            p0, p1 = pts[idx - 1], pts[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
            new_pt = (1 - alpha) * p0 + alpha * p1
            new_pts.append(new_pt)
    return np.array(new_pts, dtype=np.int32).reshape(-1, 1, 2)

class Settings:
    ROI_PERCENTAGE = 0.025
    MIN_CONTOUR_AREA = 4
    CONFIDENCE_THRESHOLD = 0.7
    TARGET_NORM_SIZE = (20, 32)
    MORPH_KERNEL_MAX_SIZE = 2
    DEFAULT_REAL_WIDTH = 10.0
    DEFAULT_REAL_HEIGHT = 2.0
    SCAN_NUMBER_MIN = 1
    SCAN_NUMBER_MAX = 99
    DELAUNAY_ALPHA = 50.0
    CONTOUR_APPROX_RATE = 0.0001
    VOLUME_DIVIDER = 1000.0
    TEMPLATES_DIR = "templates"
    ARROW_HSV_LOWER = [28, 16, 165]
    ARROW_HSV_UPPER = [36, 255, 255]
    ARROW_MIN_CONTOUR_AREA = 20
    ARROW_SYMMETRY_EPSILON = 1e-2
    NUMBER_BIN_THRESH = 200
    NUMBER_ROI_PERCENT = 0.2
    MORPH_DILATE_ITER = 2
    MORPH_ERODE_ITER = 1
    CONTOUR_HSV_LOWER = [2, 29, 145]
    CONTOUR_HSV_UPPER = [59, 255, 255]
    SATURATION_THRESHOLD = 24
    HUE_MAX = 62
    CONTOUR_MIN_POINTS = 4
    ARROW_MIN_CONTOUR_POINTS = 10
    MIN_CONTOUR_POINTS = CONTOUR_MIN_POINTS

    @classmethod
    def save(cls, path="settings.json"):
        import json
        d = {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v) and isinstance(v, (int, float, bool, str, list, tuple))}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path="settings.json"):
        import json
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
            ]),
            ("Номер", [
                ("NUMBER_BIN_THRESH", "Порог бинаризации номера", int),
                ("NUMBER_ROI_PERCENT", "ROI номера (% от размера)", float),
            ]),
            ("Морфология", [
                ("MORPH_KERNEL_MAX_SIZE", "Макс. размер ядра", int),
                ("MORPH_DILATE_ITER", "Итераций дилатации", int),
                ("MORPH_ERODE_ITER", "Итераций эрозии", int),
            ]),
            ("HSV фильтр", [
                ("CONTOUR_HSV_LOWER", "HSV-низ контура (через запятую)", list),
                ("CONTOUR_HSV_UPPER", "HSV-верх контура (через запятую)", list),
            ]),
            ("3D/Контуры", [
                ("DELAUNAY_ALPHA", "Delaunay Alpha", float),
                ("CONTOUR_APPROX_RATE", "Коэф. аппроксимации", float),
                ("VOLUME_DIVIDER", "Делитель объёма (мм³ в мл)", float),
            ]),
            ("Прочее", [
                ("SATURATION_THRESHOLD", "Порог насыщенности", int),
                ("HUE_MAX", "Максимальный оттенок (Hue Max)", int),
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
        import importlib
        import sys
        mod = sys.modules[Settings.__module__]
        importlib.reload(mod)
        self.load_settings()

@dataclass
class ModelSettings:
    real_width: float = Settings.DEFAULT_REAL_WIDTH
    real_height: float = Settings.DEFAULT_REAL_HEIGHT
    scale_x: float = 1.0
    scale_y: float = 1.0
    resample_points: int = 100
    delaunay_alpha: float = Settings.DELAUNAY_ALPHA

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
            logging.error(f"Ошибка чтения изображения {path}: {str(e)}")
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
            show_error(f"Ошибка поиска контура числа: {str(e)}")
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
            show_error(f"Ошибка нормализации числа: {str(e)}")
            return None

    def _find_arrow_roi(self, img_bgr):
        h, w = img_bgr.shape[:2]
        roi_h = int(h * 0.15)
        roi_w = int(w * 0.15)
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
            import traceback
            show_error(f"Критическая ошибка извлечения угла стрелки: {str(e)}\n{traceback.format_exc()}")
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
        best_digit, best_val = None, -1.0
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
            return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
        except Exception as e:
            show_error(f"Ошибка извлечения номера: {str(e)}")
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
            valid_by_number = [d for d in image_data if d['number'] is not None and Settings.SCAN_NUMBER_MIN <= d['number'] <= Settings.SCAN_NUMBER_MAX]
            valid_by_angle = [d for d in image_data if d['angle'] is not None]
            use_number = len(valid_by_number) == len(image_data) and len(set(d['number'] for d in valid_by_number)) == len(valid_by_number)
            use_angle = len(valid_by_angle) == len(image_data) and len(set(d['angle'] for d in valid_by_angle)) == len(valid_by_angle)
            if use_number:
                sorted_data = sorted(image_data, key=lambda d: d['number'])
            elif use_angle:
                sorted_data = sorted(image_data, key=lambda d: d['angle'])
                show_error("Сортировка по углам стрелок, номера сканов не используются или не уникальны", level="warning")
            else:
                raise ValueError("Не удалось однозначно определить порядок сканов")
            if not sorted_data:
                raise ValueError("Не найдено ни одного валидного изображения")
            self.image_files = [d['file'] for d in sorted_data]
            return [d['img'] for d in sorted_data], [d['angle'] for d in sorted_data], [d['number'] for d in sorted_data], image_shape
        except Exception as e:
            import traceback
            show_error(f"Ошибка чтения данных: {str(e)}\n{traceback.format_exc()}")
            raise

class ImageProcessor:
    def __init__(self, saturation_threshold=Settings.SATURATION_THRESHOLD, hue_max=Settings.HUE_MAX):
        self.saturation_threshold = saturation_threshold
        self.hue_max = hue_max

    @staticmethod
    def process_image(img: np.ndarray, approximation_rate: float = Settings.CONTOUR_APPROX_RATE) -> np.ndarray:
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array(Settings.CONTOUR_HSV_LOWER, dtype=np.uint8)
            upper = np.array(Settings.CONTOUR_HSV_UPPER, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            h, w = mask.shape
            kernel_size = min(Settings.MORPH_KERNEL_MAX_SIZE, h, w)
            if kernel_size < 1:
                show_error("Размер ядра морфологии слишком мал")
                return None
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=Settings.MORPH_DILATE_ITER)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if len(contour) < Settings.CONTOUR_MIN_POINTS:
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
    def __init__(self, image_width, image_height, real_width=Settings.DEFAULT_REAL_WIDTH, real_height=Settings.DEFAULT_REAL_HEIGHT, n_resample_points=120):
        self.settings = ModelSettings(
            real_width=real_width,
            real_height=real_height,
            scale_x=image_width / real_width if real_width else 1.0,
            scale_y=image_height / real_height if real_height else 1.0,
            resample_points=n_resample_points
        )
        self.points = None
        self.mesh = None
        self.volume = 0.0
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

    def set_scale(self, scale_x: float, scale_y: float):
        self.scale_x = scale_x if scale_x > 0 else self.IMAGE_WIDTH / self.REAL_WIDTH
        self.scale_y = scale_y if scale_y > 0 else self.IMAGE_HEIGHT / self.REAL_HEIGHT
        self.settings = replace(
            self.settings,
            scale_x=self.scale_x,
            scale_y=self.scale_y
        )
        logging.info(
            f"Scale set: X,Z={self.scale_x:.2f}, Y={self.scale_y:.2f} pixels/mm"
        )

    def build_model(self, contours, scan_numbers, angles=None, center=None, delaunay_alpha=None):
        try:
            if not contours or not scan_numbers:
                raise ValueError("Нет валидных контуров или номеров сканов")
            if angles is None:
                N = len(scan_numbers)
                if N == 0 or 180 % N != 0:
                    raise ValueError(f"Количество кадров ({N}) не делит 180 нацело")
                angles = [i * (180.0 / N) for i in range(N)]
            current_contours = [
                resample_contour(c, self.n_resample_points)
                for c in contours
            ]
            n = len(current_contours)
            if n > 1:
                mixed_contours = []
                for i, contour in enumerate(current_contours):
                    weighted_sum = np.zeros_like(contour.squeeze(), dtype=np.float64)
                    weights = []
                    for k in range(-n, n + 1):
                        idx = (i + k) % n
                        neighbor = current_contours[idx].squeeze()
                        weight = 1.0 / (abs(k) + 1.5) if k != 0 else 0.4
                        weighted_sum += neighbor * weight
                        weights.append(weight)
                    mixed = (
                        weighted_sum / sum(weights)
                    ).astype(np.int32).reshape(-1, 1, 2)
                    mixed_contours.append(mixed)
                current_contours = mixed_contours
            if center is None:
                center = (
                    self.IMAGE_WIDTH // 2,
                    self.IMAGE_HEIGHT // 2
                )
            contours_as_3d_points = []
            for contour in current_contours:
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
            current_delaunay_alpha = (
                delaunay_alpha if delaunay_alpha is not None else Settings.DELAUNAY_ALPHA
            )
            cloud = pv.PolyData(points)
            mesh = cloud.delaunay_3d(alpha=current_delaunay_alpha)
            surf = mesh.extract_geometry()
            logging.info(
                f"Full mesh: {surf.n_faces} faces, volume: {surf.volume:.2f}"
            )
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
        self.images = self.angles = self.scan_numbers = self.contours = self.image_files = []
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
        self.details_label = QtWidgets.QLabel("Детали распознавания:")
        self.details_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.details_label)
        help_text = ("Подсказки: ←/→ навигация, колесико мыши - масштаб, 0 - сброс масштаба, "
                     "Esc - закрыть, перетаскивание мышью - перемещение")
        help_label = QtWidgets.QLabel(help_text)
        help_label.setStyleSheet("font-size: 10px; color: gray; font-style: italic;")
        layout.addWidget(help_label)

    def set_data(self, images, angles, scan_numbers, contours, image_files):
        self.images, self.angles, self.scan_numbers, self.contours, self.image_files = images, angles, scan_numbers, contours, image_files
        self.current_index = 0
        if self.images:
            self.update_navigation_buttons()
            self.show_current_image()

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.images) - 1)

    def show_current_image(self):
        if not self.images or self.current_index >= len(self.images):
            return
        img = self.images[self.current_index]
        angle = self.angles[self.current_index] if self.current_index < len(self.angles) else "N/A"
        scan_number = self.scan_numbers[self.current_index] if self.current_index < len(self.scan_numbers) else "N/A"
        contour = self.contours[self.current_index] if self.current_index < len(self.contours) else None
        file_name = Path(self.image_files[self.current_index]).name if self.current_index < len(self.image_files) else "unknown"
        display_img = img.copy()
        h, w = img.shape[:2]
        if contour is not None:
            n_contours = len(self.contours)
            colors = plt.cm.hsv(np.linspace(0, 1, n_contours, endpoint=False))[:, :3]
            color = tuple(int(c * 255) for c in colors[self.current_index % n_contours])
            cv2.drawContours(display_img, [contour], -1, color, 2)
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(display_img_rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.info_label.setText(f"Изображение {self.current_index + 1}/{len(self.images)} | Номер скана: {scan_number} | Угол стрелки: {angle}° | Файл: {file_name}")
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
        details.append(f"Распознан номер: {scan_number}" if scan_number != "N/A" else "Номер не распознан")
        details.append(f"Распознан угол стрелки: {angle}°" if angle != "N/A" else "Угол стрелки не распознан")
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
        self.resample_points = 100
        self.delaunay_alpha = Settings.DELAUNAY_ALPHA
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
                                    "3D Scan Processor\n\n"
                                    "Программа для обработки 3D сканов и расчета объема\n"
                                    "Использует PyVista для 3D визуализации\n\n"
                                    "Версия 1.0")

    def _set_progress(self, visible: bool, maximum: int = 100, value: int = 0, text: str = ""):
        self.progress_bar.setVisible(visible)
        if text:
            self.progress_bar.setFormat(text)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()

    def _is_mesh_closed_and_manifold(self, mesh):
        try:
            return mesh.is_manifold and mesh.n_open_edges == 0 and mesh.n_faces > 0 and mesh.volume > 0
        except Exception as e:
            logging.error(f"Ошибка проверки замкнутости mesh: {e}")
            return False

    def _find_optimal_alpha(self, contours, scan_numbers, angles, initial_delaunay_alpha):
        logging.info("Starting optimal DELAUNAY_ALPHA search for closed manifold mesh...")
        low_alpha = 1.0
        high_alpha = 1000.0
        best_alpha = initial_delaunay_alpha
        max_iterations = 10
        self._set_progress(True, max_iterations, 0, "Поиск Alpha: %p%")
        for i in range(max_iterations):
            current_alpha = (low_alpha + high_alpha) / 2.0
            try:
                model = self.builder.build_model(contours, scan_numbers, angles=angles, delaunay_alpha=current_alpha)
                if self._is_mesh_closed_and_manifold(model):
                    best_alpha = current_alpha
                    high_alpha = current_alpha
                else:
                    low_alpha = current_alpha
            except Exception as e:
                logging.warning(f"Search: Failed to build model for Alpha={current_alpha:.2f}: {e}")
                low_alpha = current_alpha
            self._set_progress(True, max_iterations, i + 1)
            if abs(high_alpha - low_alpha) < 1e-2:
                break
        self._set_progress(False)
        logging.info(f"Минимальный замкнутый DELAUNAY_ALPHA найден: {best_alpha:.2f}")
        return best_alpha

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
            N = len(images)
            if N == 0 or 180 % N != 0:
                raise ValueError(f"Количество кадров ({N}) не делит 180 нацело")
            angles = [i * (180 // N) for i in range(N)]
            self.builder = ModelBuilder(image_width, image_height)
            self._set_progress(True, N, 0, "Обработка изображений: %p%")
            contours = [ImageProcessor.process_image(img) for idx, img in enumerate(images)]
            self._set_progress(False)
            if not contours:
                raise ValueError("Не удалось извлечь ни одного контура")
            self.last_contours = contours
            self.last_arrow_angles = arrow_angles
            self.last_scan_numbers = scan_numbers
            self.last_angles = angles
            self.last_images = images
            self.last_image_files = [f.name for f in self.reader.image_files]
            optimal_alpha = self._find_optimal_alpha(contours, scan_numbers, angles, Settings.DELAUNAY_ALPHA)
            model = self.builder.build_model(contours, scan_numbers, angles=angles, delaunay_alpha=optimal_alpha)
            self.visualize_model(model)
            volume_mm3 = self.builder.volume
            volume_ml = volume_mm3 / Settings.VOLUME_DIVIDER
            self.volume_label.setText(f"Объём: {volume_mm3:.4f} мм³ ({volume_ml:.5f} мл)")
        except Exception as e:
            show_error(f"Ошибка обработки: {str(e)}")
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)

    def visualize_model(self, mesh):
        try:
            if self.plotter is not None:
                self.plotter.clear()
            else:
                self.plotter = self.vtk_widget
            points = self.builder.points
            self.plotter.add_points(points, color="lightgreen", point_size=2, render_points_as_spheres=True, name="points")
            self.plotter.add_mesh(mesh, color="darkred", opacity=0.1, name="fill", lighting=False)
            self.plotter.add_mesh(mesh, color="white", opacity=0.3, style="wireframe", name="wire", line_width=0.75)
            if hasattr(self.builder, "individual_contour_3d_points") and hasattr(self.builder, "angles"):
                groups = self.builder.individual_contour_3d_points
                angles = self.builder.angles
                n_contours = len(groups)
                colors = plt.cm.hsv(np.linspace(0, 1, n_contours, endpoint=False))[:, :3]
                for i, group_points_raw in enumerate(groups):
                    if i >= len(angles):
                        continue
                    angle_rad = angles[i] * np.pi / 180
                    x_3d = group_points_raw[:, 0] * np.cos(angle_rad)
                    y_3d = group_points_raw[:, 1]
                    z_3d = group_points_raw[:, 0] * np.sin(angle_rad)
                    rotated = np.stack([x_3d, y_3d, z_3d], axis=-1)
                    if len(rotated) > 1:
                        N = len(rotated)
                        lines = np.hstack([[N] + list(range(N))])
                        poly = pv.PolyData(rotated)
                        poly.lines = lines
                        color = tuple((colors[i] * 255).astype(int))
                        self.plotter.add_mesh(poly, color=color, line_width=3, name=f"scanline_{i}")
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
        self.debug_viewer.set_data(self.last_images, self.last_arrow_angles, self.last_scan_numbers, self.last_contours, self.last_image_files)
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