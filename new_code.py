import logging
import math
import re
from math import lgamma
import traceback
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QTextEdit
from PyQt6.QtWidgets import QGraphicsView, QMessageBox
from pyvistaqt import QtInteractor
import json
from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import skimage.morphology as morphology


LOG_FILENAME = "scan_processor.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
log_file_handler = logging.FileHandler(LOG_FILENAME, mode="a", encoding="utf-8")
logging.basicConfig(handlers=[log_file_handler], level=LOG_LEVEL, format=LOG_FORMAT)


@dataclass
class ErrorEntry:
    """Структура для записи ошибки или предупреждения"""
    type: str  # "error" или "warning"
    error_type: str  # Тип ошибки (например, "InvalidImageFormat")
    file_path: str  # Путь к файлу или "N/A" если не применимо
    message: str  # Подробное сообщение
    details: str = ""  # Дополнительные детали (traceback и т.д.)


class ErrorCollector:
    """Класс для сбора ошибок и предупреждений во время обработки"""
    
    def __init__(self):
        self.errors: List[ErrorEntry] = []
        self.processed_files = 0
        self.total_files = 0
    
    def add_error(self, error_type: str, file_path: str, message: str, details: str = ""):
        """Добавить ошибку"""
        self.errors.append(ErrorEntry(
            type="error",
            error_type=error_type,
            file_path=file_path,
            message=message,
            details=details
        ))
        logging.error(f"[{error_type}] {file_path}: {message}")
    
    def add_warning(self, error_type: str, file_path: str, message: str, details: str = ""):
        """Добавить предупреждение"""
        self.errors.append(ErrorEntry(
            type="warning",
            error_type=error_type,
            file_path=file_path,
            message=message,
            details=details
        ))
        logging.warning(f"[{error_type}] {file_path}: {message}")
    
    def set_file_counts(self, processed: int, total: int):
        """Установить количество обработанных и общих файлов"""
        self.processed_files = processed
        self.total_files = total
    
    def get_summary(self) -> str:
        """Получить краткую сводку"""
        error_count = len([e for e in self.errors if e.type == "error"])
        warning_count = len([e for e in self.errors if e.type == "warning"])
        
        summary = f"Обработка завершена: {self.processed_files}/{self.total_files} файлов"
        if error_count > 0 or warning_count > 0:
            summary += f", {error_count} ошибок, {warning_count} предупреждений"
        else:
            summary += ", ошибок не обнаружено"
        
        return summary
    
    def get_grouped_report(self) -> str:
        """Получить группированный отчёт по типам ошибок"""
        if not self.errors:
            return ""
        
        # Группируем по типу ошибки
        grouped = defaultdict(list)
        for error in self.errors:
            grouped[error.error_type].append(error)
        
        report_lines = []
        for error_type, errors in grouped.items():
            count = len(errors)
            report_lines.append(f"[{error_type}] ({count} случаев):")
            
            for error in errors:
                file_display = error.file_path if error.file_path != "N/A" else "общая ошибка"
                report_lines.append(f"  - {file_display}: {error.message}")
                if error.details:
                    report_lines.append(f"    Детали: {error.details}")
        
        return "\n".join(report_lines)
    
    def show_final_report(self, parent=None):
        """Показать финальный отчёт в окне PyQt (если есть ошибки)"""
        if not self.errors:
            return
        
        # Выводим сводку в консоль
        print(self.get_summary())
        print(self.get_grouped_report())
        
        # Показываем окно только если есть ошибки
        app = QApplication.instance()
        if app:
            dlg = ErrorReportDialog(self, parent)
            dlg.exec()
    
    def clear(self):
        """Очистить все собранные ошибки"""
        self.errors.clear()
        self.processed_files = 0
        self.total_files = 0


class ErrorReportDialog(QtWidgets.QDialog):
    """Диалог для отображения финального отчёта об ошибках"""
    
    def __init__(self, error_collector: ErrorCollector, parent=None):
        super().__init__(parent)
        self.error_collector = error_collector
        self.setWindowTitle("Отчёт об обработке")
        self.setMinimumSize(600, 400)
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Сводка
        summary_label = QtWidgets.QLabel(self.error_collector.get_summary())
        summary_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        layout.addWidget(summary_label)
        
        # Отчёт
        report_text = QtWidgets.QTextEdit()
        report_text.setReadOnly(True)
        report_text.setPlainText(self.error_collector.get_grouped_report())
        layout.addWidget(report_text)
        
        # Кнопка закрытия
        btn = QtWidgets.QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# Глобальный экземпляр ErrorCollector для использования в коде
_global_error_collector = ErrorCollector()

def get_error_collector() -> ErrorCollector:
    """Получить глобальный экземпляр ErrorCollector"""
    return _global_error_collector

def reset_error_collector():
    """Сбросить глобальный ErrorCollector"""
    global _global_error_collector
    _global_error_collector.clear()


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
    MIN_CONTOUR_AREA = 40
    CONFIDENCE_THRESHOLD = 0.01
    TARGET_NORM_SIZE = (20, 32)
    MORPH_KERNEL_MAX_SIZE = 10
    MORPH_KERNEL_LABEL_SIZE = 50
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
    MORPH_DILATE_ITER = 1
    MORPH_ERODE_KERNEL_DIV_W = 4
    MORPH_ERODE_KERNEL_DIV_H = 4
    MORPH_ERODE_EXTRA_ITERATIONS = 1
    CONTOUR_HSV_LOWER = [2, 11, 0]
    CONTOUR_HSV_UPPER = [25, 255, 255]
    LABEL_HSV_LOWER = [39, 111, 0]
    LABEL_HSV_UPPER = [66, 255, 255]
    SATURATION_THRESHOLD = 24
    ARROW_MIN_CONTOUR_POINTS = 10
    MIN_CONTOUR_POINTS = 4
    MIN_ANGLE_BETWEEN_CONTOURS = 1.25
    OBJECT_TRACKING_MAX_DISTANCE_RATIO = 2.25

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
                ("OBJECT_TRACKING_MAX_DISTANCE_RATIO", "Порог дистанции трекинга (доля)", float),
            ]),
            ("Прочее", [
                ("SATURATION_THRESHOLD", "Порог насыщенности", int),
                ("TEMPLATES_DIR", "Папка шаблонов", str),
                ("MIN_CONTOUR_AREA", "Мин. площадь контура (пикс)", int),
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
            try:
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
            except ValueError:
                get_error_collector().add_warning("InvalidSettingFormat", "N/A", f"Неверный формат для настройки '{key}': '{inp.text()}'")
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
            get_error_collector().add_error("TemplateLoadFailed", "N/A", "Не удалось загрузить ни один шаблон цифры")
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
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("ImageReadError", str(path), f"Ошибка чтения изображения: {str(e)}", tb)
            return None

    def _find_number_bbox(self, gray_roi):
        try:
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Используем MIN_CONTOUR_AREA из настроек, но для цифр он может быть меньше
            valid = [c for c in contours if cv2.contourArea(c) > 4]
            if not valid:
                get_error_collector().add_warning("NoValidContours", "N/A", "Валидные контуры числа не найдены")
                return None
            all_points = np.vstack(valid).squeeze()
            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)
            return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("ContourDetectionError", "N/A", f"Ошибка поиска контура числа: {str(e)}", tb)
            return None

    def _extract_and_normalize_number(self, gray_roi, bbox):
        try:
            if bbox is None:
                return None
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                get_error_collector().add_error("InvalidBboxSize", "N/A", f"Некорректные размеры bbox: w={w}, h={h}")
                return None
            number_img = gray_roi[y:y + h, x:x + w]
            if number_img.size == 0:
                get_error_collector().add_error("EmptyImage", "N/A", "Пустое изображение числа после вырезки bbox")
                return None
            return cv2.resize(number_img, Settings.TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("NumberNormalizationError", "N/A", f"Ошибка нормализации числа: {str(e)}", tb)
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
                get_error_collector().add_warning("ArrowNotFound", "N/A", "Стрелка не найдена в ROI (контуры отсутствуют)")
                return None, None, None, None, None
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < Settings.ARROW_MIN_CONTOUR_POINTS or cv2.contourArea(contour) < Settings.ARROW_MIN_CONTOUR_AREA:
                get_error_collector().add_warning("ArrowTooSmall", "N/A", "Контур стрелки слишком мал или является шумом")
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
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("ArrowExtractionError", "N/A", f"Критическая ошибка извлечения угла стрелки: {str(e)}", tb)
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
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 4]
        bboxes = sorted(bboxes, key=lambda b: b[0])
        return [cv2.resize(roi_gray[y:y+h, x:x+w], Settings.TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA) for x, y, w, h in bboxes]

    def _recognize_digit(self, digit_img):
        if not self.digit_templates:
            get_error_collector().add_warning("NoTemplates", "N/A", "Нет шаблонов для распознавания")
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
            get_error_collector().add_warning("DigitNotRecognized", "N/A", f"Цифра не распознана, уверенность: {best_val:.2f}")
            return None, best_val
        return best_digit, best_val

    def _extract_number_from_filename(self, file_path):
        """Извлекает номер из имени файла: число перед '_' в конце (e.g., 'scan_12' → 12) или первое число в имени.
        Возвращает int в диапазоне или None."""
        try:
            filename = file_path.stem  # Без расширения
            # Приоритет: число перед '_' в конце (правильный формат, e.g., "_12")
            match = re.search(r'_(\d+)$', filename)  # $ — конец строки
            if match:
                num = int(match.group(1))
                if Settings.SCAN_NUMBER_MIN <= num <= Settings.SCAN_NUMBER_MAX:
                    return num
                else:
                    get_error_collector().add_warning("NumberOutOfRange", str(file_path), f"Номер из файла {filename} вне диапазона: {num}")
                    return None
            # Fallback: первое число в имени (если нет '_число')
            match = re.search(r'\d+', filename)
            if match:
                num = int(match.group(0))
                if Settings.SCAN_NUMBER_MIN <= num <= Settings.SCAN_NUMBER_MAX:
                    return num
                else:
                    get_error_collector().add_warning("NumberOutOfRange", str(file_path), f"Номер из файла {filename} вне диапазона: {num}")
                    return None
            return None
        except ValueError as e:
            get_error_collector().add_warning("FilenameParseError", str(file_path), f"Не удалось извлечь число из имени файла {file_path.name}: {str(e)}")
            return None

    def _extract_number(self, roi_gray):
        """Распознает номер из ROI с помощью OCR. Возвращает int или None."""
        try:
            digit_imgs = self._extract_digits_from_roi(roi_gray)
            # Оптимизация: распознаем один раз
            recognized = [self._recognize_digit(img) for img in digit_imgs]
            digits = [rec[0] for rec in recognized if rec[0] is not None]
            
            if not digits:
                return None
            
            if len(digits) == 1:
                return digits[0]
            elif len(digits) == 2:
                return digits[0] * 10 + digits[1]
            else:
                get_error_collector().add_warning("TooManyDigits", "N/A", f"Распознано {len(digits)} цифр, ожидалось 1 или 2. Используется только первая цифра.")
                return digits[0]
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("NumberExtractionError", "N/A", f"Ошибка извлечения номера: {str(e)}", tb)
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
                    get_error_collector().add_error("ImageResolutionMismatch", str(file_path), f"Обнаружено изображение с другим разрешением: {file_path.name}")
                    continue
                
                # Приоритет 1: номер из имени файла
                number = self._extract_number_from_filename(file_path)
                source = "filename"
                
                # Приоритет 2: если нет из файла, пробуем OCR
                if number is None:
                    roi_num = self._find_number_roi(img_bgr)
                    number = self._extract_number(roi_num)
                    source = "OCR"
                
                # Приоритет 3: если и OCR не дал, используем угол как "номер"
                if number is None:
                    roi_arrow = self._find_arrow_roi(img_bgr)
                    angle = self._extract_arrow_angle(roi_arrow)[0]
                    if angle is not None:
                        number = angle  # Угол как прокси для сортировки
                        source = "angle"
                    else:
                        get_error_collector().add_warning("NumberDetectionFailed", str(file_path), f"Не удалось определить номер для {file_path.name} (ни файл, ни OCR, ни угол)")
                        continue
                
                # Извлекаем угол независимо (для возврата, но не для сортировки)
                roi_arrow = self._find_arrow_roi(img_bgr)
                angle = self._extract_arrow_angle(roi_arrow)[0]
                
                image_data.append({'img': img_bgr, 'angle': angle, 'number': number, 'file': file_path, 'source': source})
                logging.info(f"Для {file_path.name} использован номер {number} из {source}")
            
            # Проверка валидности финальных номеров (уникальность)
            numbers = [d['number'] for d in image_data]
            are_numbers_valid = all(n is not None for n in numbers) and len(set(numbers)) == len(numbers)
            if not are_numbers_valid:
                get_error_collector().add_error("InvalidNumberSequence", "N/A", "Не удалось однозначно определить уникальные номера даже после fallback'ов.")
                raise ValueError("Не удалось однозначно определить уникальные номера даже после fallback'ов.")
            
            sorted_data = sorted(image_data, key=lambda d: d['number'])
            logging.info(f"Данные отсортированы по номерам: {[d['number'] for d in sorted_data]} (источники: {[d['source'] for d in sorted_data]})")
            
            if not sorted_data:
                raise ValueError("Не найдено ни одного валидного изображения для сортировки.")
            
            self.image_files = [d['file'] for d in sorted_data]
            return [d['img'] for d in sorted_data], [d['angle'] for d in sorted_data], [d['number'] for d in sorted_data], image_shape
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("DataReadingError", "N/A", f"Ошибка чтения данных: {str(e)}", tb)
            raise


class ImageProcessor:
    def __init__(self, saturation_threshold=Settings.SATURATION_THRESHOLD):
        self.saturation_threshold = saturation_threshold

    def _create_hsv_contour_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        lower = np.array(Settings.CONTOUR_HSV_LOWER, dtype=np.uint8)
        upper = np.array(Settings.CONTOUR_HSV_UPPER, dtype=np.uint8)
        return cv2.inRange(hsv_image, lower, upper)

    def _create_hsv_label_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        lower = np.array(Settings.LABEL_HSV_LOWER, dtype=np.uint8)
        upper = np.array(Settings.LABEL_HSV_UPPER, dtype=np.uint8)
        return cv2.inRange(hsv_image, lower, upper)

    def _apply_morphology(self, mask: np.ndarray, image_shape: Tuple[int, int], dilate_size=None, erode=True) -> np.ndarray:
        dilate_size = dilate_size or Settings.MORPH_KERNEL_MAX_SIZE
        h, w = image_shape[:2]
        if erode:
            kernel_circle_small = np.ones((2, 2), np.uint8)
            mask = cv2.erode(
                mask,
                kernel_circle_small,
                iterations=Settings.MORPH_ERODE_EXTRA_ITERATIONS
            )
        kernel_size = min(dilate_size, h, w)
        if kernel_size < 1:
            get_error_collector().add_warning("MorphologyKernelTooSmall", "N/A", "Размер ядра морфологии слишком мал")
            return mask
        kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel_circle, iterations=Settings.MORPH_DILATE_ITER)
        
        
        return mask

    def is_thin_contour(self, contour, threshold=0.1):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        
        if perimeter == 0:
            return False  # избегаем деления на ноль
        
        # Коэффициент компактности
        compactness = area / perimeter ** 2
        
        return compactness < threshold

    def get_neighbors(self, i, j, mask):
        """
        Возвращает список координат белых соседей (8-соседство) для пикселя (i, j).
        """
        H, W = mask.shape
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and mask[ni, nj] == 255:
                    neighbors.append((ni, nj))
        return neighbors
    
    def find_other_end(self, e_i, e_j, mask):
        """
        Находит другой конец непрерывной цепочки от endpoint (e_i, e_j).
        Возвращает координаты другого endpoint или None, если ветвление или цикл.
        """
        neighbors = self.get_neighbors(e_i, e_j, mask)
        if len(neighbors) != 1:
            return None
        s_i, s_j = neighbors[0]
        current = (s_i, s_j)
        prev = (e_i, e_j)
        while True:
            neigh = [n for n in self.get_neighbors(*current, mask) if n != prev]
            if len(neigh) == 0:
                return None  # Тупик
            if len(neigh) > 1:
                return None  # Ветвление
            next_pos = neigh[0]
            if len(self.get_neighbors(*next_pos, mask)) == 1:
                return next_pos  # Другой endpoint
            prev = current
            current = next_pos

    def repair_breaks(self, mask: np.ndarray, label_boxes: List[Tuple[int, int, int, int]],
        angle_threshold: float = 150.0) -> np.ndarray:
        """
        Функция ремонта разрывов в скелете.
        :param mask: Бинарная маска скелета (H x W, uint8, 0/255).
        :param label_boxes: Список bounding boxes (x, y, w, h) для зон разрывов.
        :param angle_threshold: Допустимая разница углов в градусах.
        :return: Отремонтированная маска.
        """

        if len(mask.shape) != 2 or mask.dtype != np.uint8:
            raise ValueError("Маска должна быть 2D uint8 массивом.")

        if not label_boxes:
            print("Предупреждение: label_boxes пустой, возвращаем оригинальную маску.")
            return mask.copy()

        working_mask = mask.copy()
        H, W = mask.shape
        avg_dim = (H + W) // 2
        N = max(3, int(avg_dim / 8.5))  # Размер окна NxN (не используется, но оставлено для совместимости)
        margin = 5  # Запас для границ маски

        def _is_inside_box(y: int, x: int, box: Tuple[int, int, int, int]) -> bool:
            """Проверяет, лежит ли точка внутри бокса."""
            x1, y1, w, h = box
            return x1 <= x < x1 + w and y1 <= y < y1 + h

        def _is_boundary_box(box: Tuple[int, int, int, int], margin: int, W: int, H: int) -> bool:
            """Проверяет, является ли бокс граничным."""
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            return x1 <= margin or x2 >= W - margin or y1 <= margin or y2 >= H - margin

        # Создание граничных боксов
        boundary_boxes = [
            (0, 0, margin, H),  # left
            (W - margin, 0, margin, H),  # right
            (0, 0, W, margin),  # top
            (0, H - margin, W, margin),  # bottom
        ]
        extended_boxes = label_boxes + boundary_boxes

        def _find_endpoints_in_boxes(m: np.ndarray, boxes: List[Tuple[int, int, int, int]], H: int, W: int, expansion_factor: float = 1.2) -> List[Dict[str, Tuple[int, int]]]:
            """Находит endpoints только внутри расширенных указанных боксов."""
            endpoints = []
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            seen_pos = set()  # Чтобы избежать дубликатов при пересекающихся боксах

            for box_index, (x1, y1, w, h) in enumerate(boxes):
                if expansion_factor <= 1.0:
                    # Без расширения
                    x_start, x_end = max(0, x1), min(W, x1 + w)
                    y_start, y_end = max(0, y1), min(H, y1 + h)
                else:
                    # Расширение симметрично от центра
                    delta_w = int(w * (expansion_factor - 1) / 2)
                    delta_h = int(h * (expansion_factor - 1) / 2)
                    x1_exp = max(0, x1 - delta_w)
                    y1_exp = max(0, y1 - delta_h)
                    w_exp = min(W - x1_exp, w + 2 * delta_w)
                    h_exp = min(H - y1_exp, h + 2 * delta_h)
                    x_start, x_end = x1_exp, x1_exp + w_exp
                    y_start, y_end = y1_exp, y1_exp + h_exp

                sub_mask = m[y_start:y_end, x_start:x_end]
                # cv2.imshow('sub', sub_mask)
                # cv2.waitKey(0)
                sub_white = np.argwhere(sub_mask == 255)

                for sy, sx in sub_white:
                    y, x = y_start + sy, x_start + sx
                    pos = (y, x)
                    if pos in seen_pos:
                        continue
                    seen_pos.add(pos)

                    # Проверка соседей в полной маске
                    neighbors = sum(
                        1 for dy, dx in offsets
                        if 0 <= y + dy < H and 0 <= x + dx < W and m[y + dy, x + dx] == 255
                    )
                    if neighbors == 1:
                        endpoints.append({'pos': pos, 'box_index': box_index})

            return endpoints

        def _get_direction_angle(m: np.ndarray, y: int, x: int, des_depth: int) -> Tuple[float, Optional[Tuple[int, int]]]:
            """Вычисляет угол направления от I-ого соседа (или max доступного) к endpoint."""
            # BFS для поиска цепочки соседей
            visited = set()
            queue = deque([(y, x, 0)])  # pos_y, pos_x, depth
            depth_map = {}
            depth_map[(y, x)] = 0
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            max_search_depth = 20  # Лимит для предотвращения полного сканирования

            while queue:
                cy, cx, depth = queue.popleft()
                if (cy, cx) in visited or depth > max_search_depth:
                    continue
                visited.add((cy, cx))
                depth_map[(cy, cx)] = depth

                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < H and 0 <= nx < W and m[ny, nx] == 255 and
                        (ny, nx) not in visited):
                        queue.append((ny, nx, depth + 1))

            # Найти соседи (depth > 0)
            neighbor_depths = {pos: d for pos, d in depth_map.items() if d > 0}
            if not neighbor_depths:
                return 0.0, None

            # Если desired_depth доступен, взять его; иначе max
            if des_depth in neighbor_depths.values():
                target_pos = next(pos for pos, d in neighbor_depths.items() if d == des_depth)
            else:
                max_d = max(neighbor_depths.values())
                target_pos = next(pos for pos, d in neighbor_depths.items() if d == max_d)

            ty, tx = target_pos
            vec = (x - tx, y - ty)  # Вектор от target к ep
            angle = np.arctan2(vec[1], vec[0]) * 180 / np.pi
            return angle, target_pos

        def _get_unconnected_candidates(active_endpoints: List[Dict], current_ep: Dict, boxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
            """Находит активных (не подключенных) endpoints в том же боксе, что и текущий."""
            box_index = current_ep['box_index']
            target_box = boxes[box_index]
            return [ep for ep in active_endpoints if not ep['connected'] and ep != current_ep and _is_inside_box(ep['pos'][0], ep['pos'][1], target_box)]

        def _select_best_candidate(m: np.ndarray, candidates: List[Dict], angle_current: float,
                                angle_threshold: float, curr_y: int, curr_x: int) -> Optional[Dict]:
            """Выбирает лучшего кандидата по углу и расстоянию. Если нет по углу - ближайший по расстоянию."""
            valid_candidates = []
            for cand in candidates:
                cand_y, cand_x = cand['pos']
                angle_cand, _ = _get_direction_angle(m, cand_y, cand_x, 2)

                # Разница углов (минимальная, 0-180)
                delta_angle = min(abs(angle_current - angle_cand), 360 - abs(angle_current - angle_cand))

                # Проверка на противоположные направления (навстречу: ~180°)
                opposite_diff = abs(delta_angle - 180)
                if opposite_diff <= angle_threshold:
                    dist = np.sqrt((curr_x - cand_x)**2 + (curr_y - cand_y)**2)
                    valid_candidates.append({'ep': cand, 'delta_angle': opposite_diff, 'dist': dist})

            if valid_candidates:
                # Сортировка: сначала по opposite_diff (меньше лучше), затем по dist
                valid_candidates.sort(key=lambda c: (c['delta_angle'], c['dist']))
                return valid_candidates[0]['ep']

            # Fallback: ближайший по расстоянию, если нет подходящих по углу
            if candidates:
                closest_cand = min(candidates, key=lambda ep: np.sqrt((curr_x - ep['pos'][1])**2 + (curr_y - ep['pos'][0])**2))
                return closest_cand

            return None

        # Шаг 1: Поиск endpoints только внутри боксов (расширенных)
        endpoints = _find_endpoints_in_boxes(working_mask, extended_boxes, H, W)

        if len(endpoints) < 2:
            return working_mask  # Нет разрывов для ремонта

        # Инициализация активных
        active_endpoints = [{**ep, 'connected': False} for ep in endpoints]

        # Обработка граничных боксов: соединение самых дальних endpoints
        boundary_boxes_to_process = [b for b in extended_boxes if _is_boundary_box(b, margin, W, H)]
        for box in boundary_boxes_to_process:
            box_endpoints = [ep for ep in active_endpoints if not ep['connected'] and _is_inside_box(ep['pos'][0], ep['pos'][1], box)]
            if len(box_endpoints) < 2:
                continue
            # Найти пару с максимальным расстоянием
            max_dist = 0
            pair = None
            for i in range(len(box_endpoints)):
                for j in range(i + 1, len(box_endpoints)):
                    py1, px1 = box_endpoints[i]['pos']
                    py2, px2 = box_endpoints[j]['pos']
                    d = np.sqrt((px1 - px2)**2 + (py1 - py2)**2)
                    if d > max_dist:
                        max_dist = d
                        pair = (box_endpoints[i], box_endpoints[j])
            if pair:
                ep1, ep2 = pair
                py1, px1 = ep1['pos']
                py2, px2 = ep2['pos']
                cv2.line(working_mask, (px1, py1), (px2, py2), 255, 1)
                ep1['connected'] = True
                ep2['connected'] = True

        # Итеративный цикл с немедленным учетом подключенных
        max_iterations = len(endpoints) // 2
        iteration = 0
        while True:
            unconnected = [ep for ep in active_endpoints if not ep['connected']]
            if len(unconnected) < 2:
                break
            connected_this_iter = False
            # Берем копию для обработки
            current_candidates = unconnected[:]
            for current_ep in current_candidates:
                if current_ep['connected']:
                    continue
                y, x = current_ep['pos']
                angle_current, _ = _get_direction_angle(working_mask, y, x, 2)

                # Шаг 2: Поиск кандидатов (не подключенные в том же боксе)
                candidates = _get_unconnected_candidates(active_endpoints, current_ep, extended_boxes)
                if not candidates:
                    continue

                # Шаг 3: Выбор лучшего
                best_cand = _select_best_candidate(working_mask, candidates, angle_current, angle_threshold, y, x)
                if best_cand:
                    cand_y, cand_x = best_cand['pos']
                    # Шаг 4: Соединение
                    cv2.line(working_mask, (x, y), (cand_x, cand_y), 255, 1)
                    current_ep['connected'] = True
                    best_cand['connected'] = True
                    connected_this_iter = True
                    # Поскольку мы фильтруем unconnected каждый раз, немедленное обновление происходит в следующей итерации while

            if not connected_this_iter:
                break
            iteration += 1
            if iteration >= max_iterations:
                break

        remaining = len([ep for ep in active_endpoints if not ep['connected']])
        if remaining > 0:
            print(f"Предупреждение: Осталось {remaining} не подключенных endpoints.")

        return working_mask

    def _find_and_filter_contours(self, idx: int, contour_mask: np.ndarray, label_boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        contour_mask = self.repair_breaks(contour_mask, label_boxes)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        result_segments = []
        for c in contours:
            if cv2.contourArea(c) > 30:
                result_segments.append(c)
        return result_segments

    def _approximate_and_resample(self, contour: np.ndarray, approximation_rate: float, n_points: int) -> np.ndarray:
        arclen = cv2.arcLength(contour, True)
        epsilon = arclen * approximation_rate
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return resample_contour(approx, n_points=n_points)

    def process_image(self, idx: int, img: np.ndarray, approximation_rate: float = Settings.CONTOUR_APPROX_RATE) -> List[np.ndarray]:
        """
        Обрабатывает изображение для извлечения всех валидных контуров.
        Возвращает: Список обработанных контуров (List[np.ndarray]).
        """
        try:
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            contour_mask = self._create_hsv_contour_mask(hsv)
            contour_mask = self._apply_morphology(contour_mask, (h, w), erode=False)

            label_mask = self._create_hsv_label_mask(hsv)
            label_kernel_size = Settings.MORPH_KERNEL_LABEL_SIZE
            label_mask = self._apply_morphology(label_mask, (h, w), label_kernel_size)
            label_contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            label_boxes = [cv2.boundingRect(c) for c in label_contours if cv2.contourArea(c) > 72]
            try:
                contour_mask = cv2.ximgproc.thinning(contour_mask)
            except cv2.error as e:
                if "function 'thinning'" in str(e):
                    get_error_collector().add_error("ThinningModuleNotFound", "N/A", 
                        "Ошибка скелетизации: модуль 'ximgproc' не найден. "
                        "Установите 'opencv-contrib-python': pip install opencv-contrib-python")
                else:
                    raise e

            all_contours = self._find_and_filter_contours(idx, contour_mask, label_boxes)

            if not all_contours:
                return []

            processed_contours = [
                self._approximate_and_resample(c, approximation_rate, n_points=150)
                for c in all_contours
            ]
            return processed_contours
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("ImageProcessingError", f"image_{idx}", f"Ошибка обработки изображения: {str(e)}", tb)
            return []


# --- ДОБАВЛЕНО: Класс для отслеживания объектов ---
@dataclass
class TrackedObject:
    id: int
    contours: Dict[int, np.ndarray] = field(default_factory=dict) # {slice_index: contour}
    original_slice_indices: Set[int] = field(default_factory=set)
    
    def get_last_contour(self) -> Tuple[int, np.ndarray]:
        if not self.contours:
            return None, None
        last_idx = max(self.contours.keys())
        return last_idx, self.contours[last_idx]


# --- ИСПРАВЛЕНО: Класс ModelBuilder с фиксом бага в интерполяции ---
class ModelBuilder:
    def __init__(self, image_width, image_height, real_width=Settings.DEFAULT_REAL_WIDTH, real_height=Settings.DEFAULT_REAL_HEIGHT, n_resample_points=250):
        self.settings = ModelSettings(
            real_width=real_width,
            real_height=real_height,
            image_width=image_width,
            image_height=image_height,
            scale_x=image_width / real_width if real_width else 1.0,
            scale_y=image_height / real_height if real_height else 1.0,
            resample_points=n_resample_points
        )
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.scale_x = self.settings.scale_x
        self.scale_y = self.settings.scale_y
        self.n_resample_points = n_resample_points
        self.tracked_objects: List[TrackedObject] = []
        # --- ВОССТАНОВЛЕНО: Поля для хранения промежуточных результатов ---
        self.points = None
        self.individual_contour_3d_points = None
        self.angles = None
        self.final_contours = None
        self.final_angles = None
        self.final_scan_numbers = None
        self.is_original_contour = None

    # --- ИЗМЕНЕНО: Добавлен параметр angles; обновлена логика отражения для больших углов ---
    def track_objects(self, all_slices_contours: List[List[np.ndarray]], original_scan_numbers: List[int], angles: List[float]):
        active_tracks: List[TrackedObject] = []
        completed_tracks: List[TrackedObject] = []
        next_obj_id = 0

        num_slices = len(all_slices_contours)
        all_contours_flat = [c for slice_contours in all_slices_contours for c in slice_contours]
        if not all_contours_flat:
            self.tracked_objects = []
            return

        avg_size = np.mean([np.sqrt(cv2.contourArea(c)) for c in all_contours_flat if cv2.contourArea(c) > 0])
        max_dist = avg_size * Settings.OBJECT_TRACKING_MAX_DISTANCE_RATIO
        logging.info(f"Трекинг объектов: средний размер = {avg_size:.1f} px, порог расстояния = {max_dist:.1f} px")

        center_y = self.IMAGE_HEIGHT / 2.0  # Для вертикального отражения

        for i in range(num_slices):
            current_contours = all_slices_contours[i]
            # --- ИЗМЕНЕНО: Правило 1: Если >2 контуров на скане — все они разные объекты ---
            if len(current_contours) > 2:
                logging.warning(f"Скан {i}: более 2 контуров ({len(current_contours)}), все считаются разными объектами.")
                # Разбиваем на отдельные "одиночные" треки, не пытаясь объединять на этом скане

            # --- ИЗМЕНЕНО: Правило 2: Проверка на разрыв (пустой скан) — сбрасываем активные треки ---
            if not current_contours and active_tracks:
                logging.info(f"Скан {i}: пустой скан (разрыв), завершаем все активные треки: {len(active_tracks)}")
                completed_tracks.extend(active_tracks)
                active_tracks = []  # Сброс: новые контуры после разрыва — новые объекты

            unmatched_contours_indices = list(range(len(current_contours)))
            
            if active_tracks and current_contours:
                # --- ИЗМЕНЕНО: Сопоставление только с ближайшим предыдущим сканом, без учёта разрывов ---
                # Вычисляем расстояния только для активных треков (каждый трек имеет ровно один контур на предыдущем скане)
                matched = set()
                for t_idx, track in enumerate(active_tracks):
                    prev_slice_idx, last_contour = track.get_last_contour()
                    if prev_slice_idx != i - 1:  # Только с соседним сканом (пропуски уже обработаны сбросом)
                        continue
                    m_prev = cv2.moments(last_contour)
                    c_prev = (m_prev['m10'] / m_prev['m00'], m_prev['m01'] / m_prev['m00']) if m_prev['m00'] != 0 else (0, 0)
                    
                    # --- ИЗМЕНЕНО: Вычисление углового расстояния и отражение, если >170° ---
                    angle_diff = abs(angles[i] - angles[prev_slice_idx])
                    angular_dist = min(angle_diff, 360 - angle_diff)
                    needs_reflection = angular_dist > 170
                    
                    best_c_idx = None
                    min_dist = np.inf
                    for c_idx, contour in enumerate(current_contours):
                        if c_idx in matched:
                            continue
                        
                        # Отражение по вертикали, если нужно
                        contour_to_use = contour
                        if needs_reflection:
                            contour_points = contour.squeeze()
                            reflected_points = np.column_stack([contour_points[:, 0], 2 * center_y - contour_points[:, 1]])
                            contour_to_use = reflected_points.reshape(-1, 1, 2)
                        
                        m_curr = cv2.moments(contour_to_use)
                        c_curr = (m_curr['m10'] / m_curr['m00'], m_curr['m01'] / m_curr['m00']) if m_curr['m00'] != 0 else (0, 0)
                        dist = np.linalg.norm(np.array(c_prev) - np.array(c_curr))
                        if dist < min_dist and dist <= max_dist:  # --- ИЗМЕНЕНО: Правило 3: Только если dist <= max_dist ---
                            min_dist = dist
                            best_c_idx = c_idx
                    
                    if best_c_idx is not None:
                        # Используем оригинальный контур (без отражения) для трека
                        contour = current_contours[best_c_idx]
                        track.contours[i] = contour
                        if original_scan_numbers[i] is not None:
                            track.original_slice_indices.add(i)
                        matched.add(best_c_idx)
                        unmatched_contours_indices.remove(best_c_idx)  # Удаляем сопоставленный
                        logging.debug(f"Трек {track.id}: сопоставлен контур {best_c_idx} на скане {i} (dist={min_dist:.1f}, reflection={needs_reflection})")

                # --- ИЗМЕНЕНО: Завершаем треки, которые не нашли пару (расстояние > max_dist) ---
                remaining_active_tracks = []
                for track in active_tracks:
                    if i in track.contours:
                        remaining_active_tracks.append(track)
                    else:
                        completed_tracks.append(track)
                        logging.debug(f"Трек {track.id}: прерван на скане {i} (расстояние слишком велико)")
                active_tracks = remaining_active_tracks

            # --- ИЗМЕНЕНО: Новые треки только для несопоставленных контуров ---
            for c_idx in unmatched_contours_indices:
                new_track = TrackedObject(id=next_obj_id)
                new_track.contours[i] = current_contours[c_idx]
                if original_scan_numbers[i] is not None:
                    new_track.original_slice_indices.add(i)
                active_tracks.append(new_track)
                next_obj_id += 1
                logging.debug(f"Новый трек {new_track.id} начат на скане {i}, контур {c_idx}")

        # Завершаем оставшиеся активные треки
        completed_tracks.extend(active_tracks)
        
        # --- ИЗМЕНЕНО: Фильтрация: только треки с >2 оригинальными срезами ---
        final_tracks = [t for t in completed_tracks if len(t.original_slice_indices) > 2]
        
        self.tracked_objects = final_tracks
        logging.info(f"Трекинг завершен. Найдено {len(final_tracks)} валидных объектов (из {next_obj_id} потенциальных).")

    # --- ИСПРАВЛЕНО: Добавлен angles в вызов track_objects ---
    def process_and_build_all_models(self, all_slices_contours, scan_numbers, angles, center=None):
        self.track_objects(all_slices_contours, scan_numbers, angles)
        
        results = []
        if not self.tracked_objects:
            get_error_collector().add_warning("NoValidObjects", "N/A", "Не найдено ни одного валидного 3D объекта после трекинга и фильтрации.")
            return []

        num_slices = len(all_slices_contours)
        
        for obj in self.tracked_objects:
            try:
                object_contour_sequence = [None] * num_slices
                object_scan_numbers = list(scan_numbers) # Копируем, чтобы не испортить оригинал
                for i in range(num_slices):
                    if i not in obj.contours:
                        object_contour_sequence[i] = None
                        object_scan_numbers[i] = -1 # Помечаем как отсутствующий для этого объекта
                    else:
                        object_contour_sequence[i] = obj.contours[i]

                prepared_contours, prepared_angles, prepared_scans, is_original = self.prepare_contours(
                    object_contour_sequence, object_scan_numbers, angles, center
                )
                
                if len(prepared_contours) < 2:
                    get_error_collector().add_warning("InsufficientSlices", f"object_{obj.id}", f"Объект ID {obj.id} имеет недостаточно срезов для построения модели. Пропуск.")
                    continue

                mesh = self.build_point_cloud_and_mesh(prepared_contours, prepared_angles, center)
                
                results.append({
                    "id": obj.id,
                    "mesh": mesh,
                    "volume_mm3": 0.0, # Будет вычислено позже
                    "final_contours": prepared_contours,
                    "final_angles": prepared_angles,
                    "final_scan_numbers": prepared_scans,
                    "is_original": is_original,
                    "viz_contours": self.individual_contour_3d_points,
                    "viz_angles": self.angles,
                })
            except Exception as e:
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                get_error_collector().add_error("ModelBuildingError", f"object_{obj.id}", f"Ошибка при построении модели для объекта ID {obj.id}: {e}", tb)
                continue
        
        return results

    # --- ВОССТАНОВЛЕН ПОЛНЫЙ ОРИГИНАЛЬНЫЙ МЕТОД ИНТЕРПОЛЯЦИИ С ОТЗЕРКАЛИВАНИЕМ ---
    def prepare_contours(self, contours, scan_numbers, angles: List[float], center=None):
        if not any(c is not None for c in contours):
            return [], [], [], []
        
        initial_contours = [
            resample_contour(c, self.n_resample_points) if c is not None else None
            for c in contours
        ]
        original_count = len(initial_contours)
        
        filled_contours, filled_angles, filled_scans = self._fill_missing_contours(
            initial_contours, list(angles), list(scan_numbers)
        )
        
        final_contours, final_angles, final_scans, is_original = self._add_missing_angles(
            filled_contours, filled_angles, filled_scans, center
        )
        
        self.is_original_contour = is_original
        self.final_contours = final_contours
        self.final_angles = final_angles
        self.final_scan_numbers = final_scans
        
        logging.info(f"Итеративная интерполяция: {original_count} исходных контуров -> {len(final_contours)} контуров")
        return final_contours, final_angles, final_scans, is_original

    # --- ИСПРАВЛЕНО: Фикс бага — интерполировать только одиночные пропуски (gap==1) ---
    def _fill_missing_contours(self, contours, angles, scan_numbers):
        n = len(contours)
        if n == 0: raise ValueError("Список контуров пуст")
        if not any(c is not None for c in contours): raise ValueError("Нет ни одного валидного контура для интерполяции")
        
        contours_list, angles_list, scans_list = list(contours), list(angles), list(scan_numbers)
        
        for idx in range(n):
            if contours_list[idx] is not None: continue
                
            prev_idx, next_idx = self._find_valid_neighbors(contours_list, idx)
            
            # ИСПРАВЛЕНО: Только если prev и next — непосредние соседи (одиночный пропуск)
            if prev_idx != -1 and next_idx != -1:
                dist_to_prev = (idx - prev_idx + n) % n
                dist_to_next = (next_idx - idx + n) % n
                if dist_to_prev == 1 and dist_to_next == 1:
                    self._interpolate_gap(contours_list, angles_list, scans_list, idx, prev_idx, next_idx)
                    continue
            
            # ИСПРАВЛЕНО: Удалено копирование от дальнего valid — оставляем None для больших gap или краёв
            # (scans_list[idx] уже -1)
        
        return contours_list, angles_list, scans_list

    def _find_valid_neighbors(self, contours, current_idx):
        n = len(contours)
        prev_idx, next_idx = -1, -1
        for i in range(1, n):
            p_idx = (current_idx - i + n) % n
            if contours[p_idx] is not None:
                prev_idx = p_idx
                break
        for i in range(1, n):
            n_idx = (current_idx + i) % n
            if contours[n_idx] is not None:
                next_idx = n_idx
                break
        return prev_idx, next_idx

    def _interpolate_gap(self, contours, angles, scans, current_idx, prev_idx, next_idx):
        n = len(contours)
        c_prev, c_next = contours[prev_idx], contours[next_idx]
        
        total_dist = (next_idx - prev_idx + n) % n
        current_dist = (current_idx - prev_idx + n) % n
        alpha = current_dist / total_dist
        
        try:
            interp = self.interpolate_contour(c_prev, c_next, alpha, linear_alpha=alpha)
        except Exception:
            interp = resample_contour(c_prev, self.n_resample_points)
        contours[current_idx] = interp
        scans[current_idx] = -1

    # --- ИСПРАВЛЕНО: Фикс — пропускать интерполяцию углов, если c1 или c2 None ---
    def _add_missing_angles(self, contours, angles, scans, center):
        n = len(contours)
        is_original = [sn is not None and sn != -1 for sn in scans]
        
        final_contours, final_angles, final_scans, final_is_original = [], [], [], []
        
        for i in range(n):
            c1, a1, s1, o1 = contours[i], angles[i], scans[i], is_original[i]
            c2, a2 = contours[(i + 1) % n], angles[(i + 1) % n]
            
            # ИСПРАВЛЕНО: Пропускаем добавление и интерполяцию, если c1 None (чтобы избежать gap)
            if c1 is None:
                continue
                
            final_contours.append(c1)
            final_angles.append(a1)
            final_scans.append(s1)
            final_is_original.append(o1)
            
            # ИСПРАВЛЕНО: Интерполируем только если c2 not None
            if c2 is not None:
                angle_diff = a2 - a1
                is_wrap = angle_diff < -90  # Переход через 180 градусов
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
                        interp_angle = (a1 + linear_alpha * angle_diff) % 180.0
                        interp_contour = self.interpolate_contour(c1, c2_for_interp, linear_alpha)
                        
                        final_contours.append(interp_contour)
                        final_angles.append(interp_angle)
                        final_scans.append(-1)
                        final_is_original.append(False)
        
        # ИСПРАВЛЕНО: Сортировка только валидных (без None)
        valid_zipped = [(ang, cont, scn, orig) for ang, cont, scn, orig in zip(final_angles, final_contours, final_scans, final_is_original) if cont is not None]
        if not valid_zipped:
            return [], [], [], []
        
        zipped = sorted(valid_zipped, key=lambda x: x[0])
        angles_out, contours_out, scans_out, is_original_out = zip(*zipped)
        
        return list(contours_out), list(angles_out), list(scans_out), list(is_original_out)

    def build_point_cloud_and_mesh(self, contours, angles, center=None):
        if center is None:
            center = (self.IMAGE_WIDTH // 2, self.IMAGE_HEIGHT // 2)
        
        contours_as_3d_points = []
        for contour in contours:
            if contour is None: continue
            current_contour_3d_points = [
                [(p[0][0] - center[0]) / self.scale_x, (center[1] - p[0][1]) / self.scale_y, 0.0]
                for p in contour
            ]
            contours_as_3d_points.append(np.array(current_contour_3d_points))

        points_list = []
        for i, contour_3d_points in enumerate(contours_as_3d_points):
            angle_rad = angles[i] * np.pi / 180
            for p in contour_3d_points:
                x_3d = p[0] * np.cos(angle_rad)
                y_3d = p[1]
                z_3d = p[0] * np.sin(angle_rad)
                points_list.append([x_3d, y_3d, z_3d])
        
        points = np.array(points_list)
        if points.shape[0] < 4:
            raise ValueError(f"Недостаточно точек для триангуляции: {points.shape[0]}")

        cloud = pv.PolyData(points)
        grid = cloud.delaunay_3d(alpha=0.05, tol=0.01, offset=2.5)
        surf = grid.extract_geometry().clean(tolerance=1e-4)
        
        self.points = points
        self.individual_contour_3d_points = contours_as_3d_points
        self.angles = angles
        
        return surf

    def interpolate_contour(self, contour1, contour2, alpha, linear_alpha=None):
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
            center_x, center_y = self.IMAGE_WIDTH / 2.0, self.IMAGE_HEIGHT / 2.0
        else:
            center_x, center_y = center[0], center[1]
        
        valid_data = [(ang, cont) for ang, cont in zip(angles, contours) if cont is not None]
        if not valid_data: return 0.0
        
        sorted_angles, sorted_contours = zip(*sorted(valid_data, key=lambda x: x[0]))

        moments_mm3 = []
        for contour_px in sorted_contours:
            points_px = contour_px.reshape(-1, 2).astype(np.float32)
            points_mm = np.zeros_like(points_px)
            points_mm[:, 0] = (points_px[:, 0] - center_x) / self.scale_x
            points_mm[:, 1] = (points_px[:, 1] - center_y) / self.scale_y
            moment = self._first_moment_of_area(points_mm)
            moments_mm3.append(moment)
        
        angles_rad = np.deg2rad(sorted_angles)
        unique_angles, unique_indices = np.unique(angles_rad, return_index=True)
        unique_moments = np.array(moments_mm3)[unique_indices]

        if len(unique_angles) < 2: return 0.0

        volume = float(np.trapz(y=unique_moments, x=unique_angles))
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

    # --- ИЗМЕНЕНО: Принимает списки списков для контуров и цветов ---
    def set_data(self, images: List[np.ndarray], scan_numbers: List[int], contours: List[List[np.ndarray]], angles: List[float], colors: List[List[Tuple[int, int, int]]], scan_to_image_map: Dict[int, int]):
        self.images = images
        self.scan_numbers = scan_numbers
        self.contours = contours # Теперь это [[c1, c2], [c3], ...]
        self.angles = angles
        self.colors = colors # Теперь это [[color1, color2], [color3], ...]
        self.scan_to_image_map = scan_to_image_map
        self.current_index = 0
        self.update_filtered_indices()
        if self.filtered_indices:
            self.show_current_image()

    def update_filtered_indices(self):
        # Теперь self.angles определяет количество кадров
        all_indices = list(range(len(self.angles)))
        if self.show_interpolated:
            self.filtered_indices = all_indices
        else:
            self.filtered_indices = [i for i in all_indices if self.scan_numbers[i] != -1]
        
        if not self.filtered_indices:
             self.current_index = 0
        elif self.current_index >= len(self.filtered_indices):
            self.current_index = 0
            
        self.prev_button.setEnabled(bool(self.filtered_indices))
        self.next_button.setEnabled(bool(self.filtered_indices))

    def on_interp_checkbox_changed(self, state):
        self.show_interpolated = bool(state)
        self.update_filtered_indices()
        self.show_current_image()

    def show_current_image(self):
        if not self.filtered_indices:
            self.info_label.setText("Нет данных для отображения")
            self.scene.clear()
            self.details_label.setText("")
            return
        
        frame_idx = self.filtered_indices[self.current_index]
        
        contours_for_frame = self.contours[frame_idx]
        colors_for_frame = self.colors[frame_idx]
        angle = self.angles[frame_idx]
        scan_number = self.scan_numbers[frame_idx]
        is_original = scan_number != -1
        
        if is_original:
            image_index = self.scan_to_image_map.get(scan_number)
            if image_index is not None and 0 <= image_index < len(self.images):
                img = self.images[image_index].copy()
            else:
                img = np.zeros_like(self.images[0], dtype=np.uint8) if self.images else np.zeros((100, 100, 3), dtype=np.uint8)
            info_text = f"Кадр {self.current_index + 1}/{len(self.filtered_indices)} (Оригинал) | Скан: {scan_number} | Угол: {angle:.2f}°"
        else:
            img = np.zeros_like(self.images[0], dtype=np.uint8) if self.images else np.zeros((100, 100, 3), dtype=np.uint8)
            info_text = f"Кадр {self.current_index + 1}/{len(self.filtered_indices)} (Интерполированный) | Угол: {angle:.2f}°"
        
        # Отрисовываем все контуры для данного кадра
        for i, contour in enumerate(contours_for_frame):
            if contour is not None and len(contour) > 0:
                color = colors_for_frame[i]
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
        
        # Вычисляем площади контуров в реальных единицах
        details_parts = [f"Найдено контуров на кадре: {len(contours_for_frame)}"]
        
        if contours_for_frame:
            # Получаем настройки масштаба из главного окна
            main_window = self.parent()
            if hasattr(main_window, 'builder') and main_window.builder:
                scale_x = main_window.builder.scale_x
                scale_y = main_window.builder.scale_y
                
                for i, contour in enumerate(contours_for_frame):
                    if contour is not None and len(contour) > 0:
                        # Вычисляем площадь в пикселях
                        area_px = cv2.contourArea(contour)
                        # Переводим в мм² с учётом масштаба
                        area_mm2 = area_px / (scale_x * scale_y)
                        details_parts.append(f"Контур {i+1}: {area_mm2:.3f} мм²")
        
        details = "\n".join(details_parts)
        self.details_label.setText(details)

    def show_previous(self):
        if not self.filtered_indices: return
        self.current_index = (self.current_index - 1 + len(self.filtered_indices)) % len(self.filtered_indices)
        self.show_current_image()

    def show_next(self):
        if not self.filtered_indices: return
        self.current_index = (self.current_index + 1) % len(self.filtered_indices)
        self.show_current_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene and not self.scene.sceneRect().isEmpty():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left: self.show_previous()
        elif key == Qt.Key.Key_Right: self.show_next()
        elif key == Qt.Key.Key_Escape: self.close()
        elif key == Qt.Key.Key_0:
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else: super().keyPressEvent(event)

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
        self.last_images = []
        self.last_debug_data = {}
        self.init_ui()

    def init_ui(self):
        self.create_menu_bar()
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        self.vtk_widget = QtInteractor(central_widget)
        layout.addWidget(self.vtk_widget.interactor)
        
        self.volume_display = QTextEdit("Объём: N/A")
        self.volume_display.setReadOnly(True)
        self.volume_display.setFixedHeight(100)
        self.volume_display.setStyleSheet("font-size: 14px; color: blue;")
        layout.addWidget(self.volume_display)
        
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
                                    "3D Scan Processor (Multi-Object)\n"
                                    "Версия 2.2\n"
                                    "Программа для обработки 3D сканов и расчета объема нескольких объектов.")

    def _set_progress(self, visible: bool, maximum: int = 100, value: int = 0, text: str = ""):
        self.progress_bar.setVisible(visible)
        if text:
            self.progress_bar.setFormat(text)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()

    def select_folder(self):
        try:
            # Сбрасываем коллектор ошибок перед новой обработкой
            reset_error_collector()
            
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
            if not folder:
                return
            self.reader.directory = Path(folder)
            images, arrow_angles, scan_numbers, image_shape = self.reader.read_images()
            print(images, scan_numbers, arrow_angles)
            if image_shape is None:
                get_error_collector().add_error("ImageResolutionError", "N/A", "Не удалось определить разрешение изображений")
                raise ValueError("Не удалось определить разрешение изображений")
            
            self.last_images = images
            image_height, image_width = image_shape[:2]
            self.scan_to_image_map = {num: i for i, num in enumerate(scan_numbers) if num is not None}
            
            if all(n is not None for n in scan_numbers):
                N = len(images)
                if N == 0: 
                    get_error_collector().add_error("NoImagesFound", "N/A", "Изображения не найдены.")
                    raise ValueError("Изображения не найдены.")
                angles = [i * (180.0 / N) for i in range(N)]
                logging.info("Порядок определен по номерам сканов, используются сгенерированные углы.")
            else:
                angles = arrow_angles
                logging.info("Порядок определен по углам, используются распознанные углы.")
            
            self.builder = ModelBuilder(image_width, image_height, n_resample_points=150)
            
            # Устанавливаем счетчики файлов для ErrorCollector
            get_error_collector().set_file_counts(len(images), len(images))
            
            self._set_progress(True, len(images), 0, "Обработка изображений: %p%")
            all_slices_contours = []
            for i, img in enumerate(images):
                contours = self.image_processor.process_image(i, img)
                all_slices_contours.append(contours)
                self._set_progress(True, len(images), i + 1)

            self._set_progress(True, 100, 50, "Трекинг и построение моделей...")
            
            results = self.builder.process_and_build_all_models(all_slices_contours, scan_numbers, angles, center=(image_width/2, image_height/2))
            
            # --- ИЗМЕНЕНО: Восстановлена оригинальная логика вычисления объёма ---
            for res in results:
                contours_list = res['final_contours']
                angles_list = res['final_angles']
                
                half_contours = []
                half_angles = []
                for contour_px, ang in zip(contours_list, angles_list):
                    if contour_px is None or len(contour_px) < 3:
                        continue
                    n_full = contour_px.shape[0]
                    n_half = n_full // 2 + 1
                    # Делим контур на две половины
                    right_half = contour_px[:n_half]
                    left_half = contour_px[n_half - 1:]
                    # Добавляем обе половины с соответствующими углами (вторая "отзеркалена")
                    half_contours.append(right_half)
                    half_angles.append(ang)
                    half_contours.append(left_half)
                    half_angles.append((ang + 180.0) % 360.0)
                
                # Вычисляем объем по полному 360-градусному набору данных
                vol_radial = self.builder.volume_radial_integration(half_contours, half_angles)
                res['volume_mm3'] = float(vol_radial)

            self._set_progress(False)

            if not results:
                self.volume_display.setText("Объекты не найдены или не удалось построить модели.")
                if self.plotter: self.plotter.clear()
                return

            num_original_scans = len(scan_numbers)
            slice_colors_map = plt.cm.hsv(np.linspace(0, 1, num_original_scans, endpoint=False))
            
            self.prepare_viz_and_debug_data(results, slice_colors_map)
            
            self.visualize_models(results, slice_colors_map)
            
            total_volume_mm3 = sum(r['volume_mm3'] for r in results)
            total_volume_ml = total_volume_mm3 / Settings.VOLUME_DIVIDER
            report_text = f"<b>Суммарный объём: {total_volume_mm3:.3f} мм³ ({total_volume_ml:.4f} мл)</b>\n"
            report_text += f"Найдено объектов: {len(results)}\n"
            report_text += "-"*30 + "\n"
            for res in sorted(results, key=lambda x: x['id']):
                vol_mm3 = res['volume_mm3']
                vol_ml = vol_mm3 / Settings.VOLUME_DIVIDER
                report_text += f"Объект ID {res['id']}: {vol_mm3:.3f} мм³ ({vol_ml:.4f} мл)\n"
            
            self.volume_display.setHtml(report_text.replace("\n", "<br>"))
            
            # Показываем финальный отчет об ошибках
            get_error_collector().show_final_report(self)

        except Exception as e:
            self._set_progress(False)
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("ProcessingError", "N/A", f"Ошибка обработки: {str(e)}", tb)
            get_error_collector().show_final_report(self)
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)

    def prepare_viz_and_debug_data(self, results, slice_colors_map):
        """Группирует все данные по кадрам (уникальным углам) для DebugViewer."""
        frames_data = defaultdict(lambda: {'contours': [], 'colors': [], 'scan_number': -1})
        gray_color_bgr = (180, 180, 180)

        # 1. Собрать все контуры в словарь, где ключ - угол среза
        for res in results:
            for i in range(len(res['final_contours'])):
                angle = res['final_angles'][i]
                contour = res['final_contours'][i]
                is_orig = res['is_original'][i]
                scan_num = res['final_scan_numbers'][i]

                color_bgr_to_use = gray_color_bgr
                if is_orig:
                    frames_data[angle]['scan_number'] = scan_num
                    if scan_num in self.scan_to_image_map:
                        original_slice_index = self.scan_to_image_map[scan_num]
                        color_np = slice_colors_map[original_slice_index] * 255
                        color_rgb = tuple(map(int, color_np[:3]))
                        color_bgr_to_use = rgb_to_bgr(color_rgb)
                
                frames_data[angle]['contours'].append(contour)
                frames_data[angle]['colors'].append(color_bgr_to_use)

        # 2. Отсортировать кадры по углу и развернуть в списки
        sorted_frames = sorted(frames_data.items(), key=lambda item: item[0])
        
        final_angles = [item[0] for item in sorted_frames]
        final_contours_per_frame = [item[1]['contours'] for item in sorted_frames]
        final_colors_per_frame = [item[1]['colors'] for item in sorted_frames]
        final_scan_numbers = [item[1]['scan_number'] for item in sorted_frames]

        self.last_debug_data = {
            "angles": final_angles,
            "contours": final_contours_per_frame, # Теперь это список списков
            "colors": final_colors_per_frame,     # И это тоже
            "scan_numbers": final_scan_numbers,
        }

    def visualize_models(self, results, slice_colors_map):
        try:
            if self.plotter is not None: self.plotter.clear()
            else: self.plotter = self.vtk_widget
            self.plotter.set_background((0.1, 0.1, 0.15))
            
            gray_color_rgb_float = (180/255, 180/255, 180/255)

            for res in results:
                groups = res['viz_contours']
                angles = res['viz_angles']
                is_original_flags = res['is_original']
                scan_numbers = res['final_scan_numbers']

                for i, group_points_raw in enumerate(groups):
                    if i >= len(angles): continue
                    
                    is_orig = is_original_flags[i]
                    color_to_use, line_width, opacity = gray_color_rgb_float, 1.5, 0.2
                    
                    if is_orig:
                        scan_num = scan_numbers[i]
                        if scan_num in self.scan_to_image_map:
                            original_slice_index = self.scan_to_image_map[scan_num]
                            color_to_use = tuple(slice_colors_map[original_slice_index][:3])
                        line_width, opacity = 3.0, 1.0

                    angle_rad = angles[i] * np.pi / 180
                    x_3d = group_points_raw[:, 0] * np.cos(angle_rad)
                    y_3d = group_points_raw[:, 1]
                    z_3d = group_points_raw[:, 0] * np.sin(angle_rad)
                    rotated = np.stack([x_3d, y_3d, z_3d], axis=-1)
                    
                    if len(rotated) > 1:
                        poly = pv.lines_from_points(rotated, close=True)
                        self.plotter.add_mesh(poly, color=color_to_use, line_width=line_width, opacity=opacity, name=f"scanline_{res['id']}_{i}")

            self.plotter.reset_camera()
            axes = pv.AxesAssembly(label_color="white", label_size=12)
            self.plotter.add_orientation_widget(axes)
            self.plotter.update()
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            get_error_collector().add_error("VisualizationError", "N/A", f"Ошибка визуализации: {str(e)}", tb)

    def open_debug_viewer(self):
        if not self.last_images or not self.last_debug_data:
            get_error_collector().add_warning("NoDebugData", "N/A", "Нет данных для отладки. Сначала выберите папку с изображениями.")
            return
        
        self.debug_viewer.set_data(
            images=self.last_images,
            scan_numbers=self.last_debug_data['scan_numbers'],
            contours=self.last_debug_data['contours'],
            angles=self.last_debug_data['angles'],
            colors=self.last_debug_data['colors'],
            scan_to_image_map=self.scan_to_image_map 
        )
        self.debug_viewer.show()
        self.debug_viewer.raise_()
        self.debug_viewer.activateWindow()

    def open_settings_dialog(self):
        dlg = SettingsDialog(self)
        dlg.exec()


if __name__ == "__main__":
    Settings.load()
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(True)
    window = MainWindow()
    window.show()
    app.exec()