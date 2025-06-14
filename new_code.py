import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from PyQt6 import QtWidgets, QtCore
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import logging
from scipy.interpolate import interp1d

# Настройка логирования
logging.basicConfig(filename='scan_processor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataReader:
    def __init__(self, directory, templates_dir="templates"):
        self.directory = Path(directory)
        self.templates_dir = Path(templates_dir)
        self.images = []
        self.scan_numbers = []
        self.templates = None
        self.ROI_PERCENTAGE = 0.05
        self.MIN_CONTOUR_AREA = 4
        self.CONFIDENCE_THRESHOLD = 0.7
        self.TARGET_NORM_SIZE = (20, 32)  # (ширина, высота)
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
            logging.error(f"Не удалось загрузить изображение: {image_path}")
            return None, None, None, None

        h_orig, w_orig = img_bgr.shape[:2]
        roi_size = int(min(h_orig, w_orig) * self.ROI_PERCENTAGE)
        x1, y1 = 0, 0
        x2, y2 = roi_size, roi_size

        roi_bgr = img_bgr[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return roi_bgr, roi_gray, img_bgr, (x1, y1, x2, y2)

    def _find_number_bbox(self, gray_roi):
        """Находит ограничивающий прямоугольник для числа."""
        try:
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                logging.warning("Контуры числа не найдены")
                return None

            valid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_CONTOUR_AREA]
            if not valid_contours:
                logging.warning("Валидные контуры числа не найдены")
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
            logging.error(f"Ошибка поиска контура числа: {str(e)}")
            return None

    def _extract_and_normalize_number(self, gray_roi, bbox):
        """Извлекает и нормализует изображение числа."""
        try:
            if bbox is None:
                return None
            x, y, w, h = bbox
            number_img = gray_roi[y:y + h, x:x + w]
            normalized_number = cv2.resize(number_img, self.TARGET_NORM_SIZE, interpolation=cv2.INTER_AREA)
            return normalized_number
        except Exception as e:
            logging.error(f"Ошибка нормализации числа: {str(e)}")
            return None

    def _load_templates(self):
        """Загружает нормализованные шаблоны чисел."""
        self.templates = {}
        for i in range(1, 19):
            template_path = self.templates_dir / f"{i}.png"
            template_bgr = self._imread_unicode(template_path)
            if template_bgr is None:
                logging.warning(f"Шаблон для числа {i} не найден: {template_path}")
                continue
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            bbox = self._find_number_bbox(template_gray)
            normalized_template = self._extract_and_normalize_number(template_gray, bbox)
            if normalized_template is not None:
                self.templates[i] = normalized_template
        if not self.templates:
            logging.error("Не удалось загрузить ни один шаблон")
            self.templates = None

    def _recognize_number(self, normalized_number_img):
        """Распознаёт число с помощью шаблонов."""
        if normalized_number_img is None or self.templates is None or not self.templates:
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
            logging.warning(f"Число не распознано, уверенность: {best_match_value:.2f}")
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
            logging.error(f"Ошибка извлечения номера: {str(e)}")
            return None

    def read_images(self):
        """Читает изображения и извлекает номера сканов."""
        try:
            self.images = []
            self.scan_numbers = []
            seen_numbers = set()  # Для отслеживания уникальных номеров
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                for file in self.directory.glob(ext):
                    _, roi_gray, img, _ = self._load_and_crop_roi(file)
                    number = self._extract_number(roi_gray)
                    if number is not None and 1 <= number <= 18:
                        if number in seen_numbers:
                            logging.warning(f"Обнаружен дубликат номера скана {number} в файле: {file}")
                            continue
                        self.images.append(img)
                        self.scan_numbers.append(number)
                        seen_numbers.add(number)
                    else:
                        logging.warning(f"Неверный номер скана в файле: {file}")

            missing = set(range(1, 19)) - set(self.scan_numbers)
            if missing:
                logging.warning(f"Отсутствуют сканы: {missing}")
            if not self.images:
                raise ValueError("Не найдено ни одного валидного изображения")
            return self.images, self.scan_numbers
        except Exception as e:
            logging.error(f"Ошибка чтения данных: {str(e)}", exc_info=True)
            raise


class ImageProcessor:
    def __init__(self, saturation_threshold=0.0):
        self.saturation_threshold = saturation_threshold

    def process_image(self, img, approximation_rate=0.0003):
        """Выделяет контур из изображения."""
        try:
            img = cv2.resize(img, (512, 512))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > (self.saturation_threshold * 255)
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if len(contour) < 4:
                    logging.warning("Контур слишком мал")
                    return None
                arclen = cv2.arcLength(contour, True)
                epsilon = arclen * approximation_rate
                approx = cv2.approxPolyDP(contour, epsilon, True)
                return approx
            else:
                logging.warning("Контур не найден")
                return None
        except Exception as e:
            logging.error(f"Ошибка обработки изображения: {str(e)}", exc_info=True)
            return None


class ModelBuilder:
    def __init__(self):
        self.points = vtk.vtkPoints()
        self.polydata = vtk.vtkPolyData()
        self.volume = 0.0
        self.scale_x = 1.0  # Пикселей на мм по X и Z
        self.scale_y = 1.0  # Пикселей на мм по Y

    def set_scale(self, scale_x, scale_y):
        """Устанавливает анизотропный масштаб для X (и Z) и Y."""
        self.scale_x = scale_x if scale_x > 0 else 1.0
        self.scale_y = scale_y if scale_y > 0 else 1.0
        logging.info(f"Установлен масштаб: X,Z={self.scale_x}, Y={self.scale_y} пикселей на мм")

    def build_model(self, contours, scan_numbers, center=(256, 256)):
        """Создаёт 3D-модель из контуров с учетом масштаба."""
        try:
            if not contours or not scan_numbers:
                raise ValueError("Нет валидных контуров или номеров сканов")

            contours, angles = self._interpolate_missing(contours, scan_numbers)

            self.points.Reset()
            for i, contour in enumerate(contours):
                angle = angles[i] * np.pi / 180
                for point in contour:
                    x, y = point[0]
                    x_centered = (x - center[0]) / self.scale_x  # Масштабируем по X
                    y_centered = (y - center[1]) / self.scale_y  # Масштабируем по Y
                    x_3d = x_centered * np.cos(angle)
                    y_3d = y_centered
                    z_3d = x_centered * np.sin(angle)  # Z наследует масштаб X
                    self.points.InsertNextPoint(x_3d, y_3d, z_3d)

            if self.points.GetNumberOfPoints() < 4:
                raise ValueError(f"Недостаточно точек для триангуляции: {self.points.GetNumberOfPoints()}")

            self.polydata.SetPoints(self.points)

            delaunay = vtk.vtkDelaunay3D()
            delaunay.SetInputData(self.polydata)
            delaunay.SetAlpha(50.0)
            delaunay.Update()

            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(delaunay.GetOutput())
            geometry_filter.Update()
            self.polydata = geometry_filter.GetOutput()

            feature_edges = vtk.vtkFeatureEdges()
            feature_edges.SetInputData(self.polydata)
            feature_edges.BoundaryEdgesOn()
            feature_edges.Update()
            if feature_edges.GetOutput().GetNumberOfCells() > 0:
                logging.warning("Поверхность не замкнута, объём может быть неточным")

            self.volume = self._calculate_volume(self.polydata)
            return self.polydata
        except Exception as e:
            logging.error(f"Ошибка построения модели: {str(e)}", exc_info=True)
            raise

    def _interpolate_missing(self, contours, scan_numbers):
        """Интерполирует недостающие и промежуточные сканы для шага 10 градусов."""
        try:
            full_angles = list(np.arange(0, 360, 10))
            full_contours = [None] * len(full_angles)

            for i, n in enumerate(scan_numbers):
                index = (n - 1) * 2
                full_contours[index] = contours[i]

            for i in range(len(full_angles)):
                if full_contours[i] is None:
                    prev_idx = next((j for j in range(i - 1, -1, -1) if full_contours[j] is not None), None)
                    next_idx = next((j for j in range(i + 1, len(full_angles)) if full_contours[j] is not None), None)
                    if prev_idx is not None and next_idx is not None:
                        t = (full_angles[i] - full_angles[prev_idx]) / (full_angles[next_idx] - full_angles[prev_idx])
                        full_contours[i] = self._interpolate_contour(full_contours[prev_idx], full_contours[next_idx],
                                                                     t)
                    else:
                        logging.warning(
                            f"Невозможно интерполировать контур для угла {full_angles[i]}: отсутствуют соседние данные")

            valid_contours = []
            valid_angles = []
            for i, contour in enumerate(full_contours):
                if contour is not None:
                    valid_contours.append(contour)
                    valid_angles.append(full_angles[i])

            if len(valid_contours) < len(full_angles):
                logging.warning(f"Создано {len(valid_contours)} контуров вместо 36")

            return valid_contours, valid_angles
        except Exception as e:
            logging.error(f"Ошибка интерполяции контуров: {str(e)}")
            return contours, [20 * (n - 1) for n in scan_numbers]

    def _interpolate_contour(self, c1, c2, t):
        """Интерполирует между двумя контурами."""
        try:
            n_points = min(len(c1), len(c2))
            c1 = cv2.approxPolyDP(c1, 1.0, True)[:n_points]
            c2 = cv2.approxPolyDP(c2, 1.0, True)[:n_points]
            return (c1 * (1 - t) + c2 * t).astype(np.int32)
        except Exception as e:
            logging.error(f"Ошибка интерполяции контура: {str(e)}")
            return c1

    def _calculate_volume(self, mesh):
        """Вычисляет объём модели."""
        try:
            mass = vtk.vtkMassProperties()
            mass.SetInputData(mesh)
            mass.Update()
            return mass.GetVolume()
        except Exception as e:
            logging.error(f"Ошибка вычисления объёма: {str(e)}")
            return 0.0

    def export(self, filename, format='ply'):
        """Экспортирует модель в указанном формате."""
        try:
            if format == 'ply':
                writer = vtk.vtkPLYWriter()
            elif format == 'xyz':
                writer = vtk.vtkXYZWriter()
            else:
                raise ValueError(f"Формат {format} не поддерживается")
            writer.SetFileName(filename)
            writer.SetInputData(self.polydata)
            writer.Write()
            logging.info(f"Модель экспортирована в {filename}")
        except Exception as e:
            logging.error(f"Ошибка экспорта: {str(e)}", exc_info=True)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Scan Processor")
        self.setGeometry(100, 100, 800, 600)
        self.reader = DataReader('.')
        self.processor = ImageProcessor()
        self.builder = ModelBuilder()
        self.point_actor = None
        self.surface_actor = None
        self.renderer = None
        self.is_point_mode = True
        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.select_button = QtWidgets.QPushButton("Выбрать папку")
        self.select_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_button)

        self.settings_button = QtWidgets.QPushButton("Настройки")
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)

        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        layout.addWidget(self.vtk_widget)

        self.volume_label = QtWidgets.QLabel("Объём: N/A")
        self.volume_label.setStyleSheet("font-size: 16px; color: blue;")
        self.volume_label.mousePressEvent = self.copy_volume
        layout.addWidget(self.volume_label)

        self.toggle_button = QtWidgets.QPushButton("Переключить на поверхность")
        self.toggle_button.clicked.connect(self.toggle_mode)
        layout.addWidget(self.toggle_button)

        self.export_button = QtWidgets.QPushButton("Экспортировать модель")
        self.export_button.clicked.connect(self.export_model)
        layout.addWidget(self.export_button)

    def select_folder(self):
        """Обработка выбора папки."""
        try:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
            if folder:
                self.reader.directory = Path(folder)
                images, scan_numbers = self.reader.read_images()
                with Pool() as pool:
                    contours = pool.map(self.processor.process_image, images)
                contours = [c for c in contours if c is not None]
                if not contours:
                    raise ValueError("Не удалось извлечь ни одного контура")
                model = self.builder.build_model(contours, scan_numbers)
                volume_mm3 = self.builder.volume
                volume_ml = volume_mm3 / 1000.0
                self.volume_label.setText(f"Объём: {volume_mm3:.2f} мм³ ({volume_ml:.2f} мл)")
                self.visualize_model(model)
        except Exception as e:
            logging.error(f"Ошибка обработки: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка обработки: {str(e)}")

    def open_settings(self):
        """Открывает окно настроек для ввода масштаба по X и Y."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки")
        layout = QtWidgets.QVBoxLayout(dialog)

        layout.addWidget(QtWidgets.QLabel("Масштаб по X и Z (пикселей на мм):"))
        scale_x_input = QtWidgets.QLineEdit()
        scale_x_input.setText(str(self.builder.scale_x))
        layout.addWidget(scale_x_input)

        layout.addWidget(QtWidgets.QLabel("Масштаб по Y (пикселей на мм):"))
        scale_y_input = QtWidgets.QLineEdit()
        scale_y_input.setText(str(self.builder.scale_y))
        layout.addWidget(scale_y_input)

        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.exec()
        try:
            scale_x = float(scale_x_input.text())
            scale_y = float(scale_y_input.text())
            self.builder.set_scale(scale_x, scale_y)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Ошибка",
                                          "Неверный формат масштаба. Используются значения по умолчанию (1.0)")
            self.builder.set_scale(1.0, 1.0)

    def visualize_model(self, polydata):
        """Визуализация модели с переключением между облаком точек и поверхностью."""
        try:
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetRadius(1.0)
            sphere_source.SetThetaResolution(8)
            sphere_source.SetPhiResolution(8)

            glyph = vtk.vtkGlyph3D()
            glyph.SetSourceConnection(sphere_source.GetOutputPort())
            glyph.SetInputData(polydata)
            glyph.ScalingOff()
            glyph.Update()

            point_mapper = vtk.vtkPolyDataMapper()
            point_mapper.SetInputConnection(glyph.GetOutputPort())
            self.point_actor = vtk.vtkActor()
            self.point_actor.SetMapper(point_mapper)
            self.point_actor.GetProperty().SetColor(1.0, 1.0, 1.0)

            surface_mapper = vtk.vtkPolyDataMapper()
            surface_mapper.SetInputData(polydata)
            self.surface_actor = vtk.vtkActor()
            self.surface_actor.SetMapper(surface_mapper)
            self.surface_actor.GetProperty().SetColor(0.8, 0.8, 0.8)

            self.renderer = vtk.vtkRenderer()
            self.renderer.AddActor(self.point_actor)
            self.renderer.AddActor(self.surface_actor)
            self.renderer.SetBackground(0.1, 0.2, 0.4)

            self.point_actor.VisibilityOn()
            self.surface_actor.VisibilityOff()

            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(100, 100, 100)
            axes.SetShaftTypeToCylinder()
            axes.SetAxisLabels(True)
            self.renderer.AddActor(axes)

            render_window = self.vtk_widget.GetRenderWindow()
            render_window.AddRenderer(self.renderer)

            interactor = self.vtk_widget
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
            style.SetMotionFactor(5.0)

            self.renderer.ResetCamera()
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, 0, 500)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)

            logging.info(f"Модель инициализирована с {polydata.GetNumberOfPoints()} точками")

            render_window.Render()
            interactor.Start()
        except Exception as e:
            logging.error(f"Ошибка визуализации: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка визуализации: {str(e)}")

    def toggle_mode(self):
        """Переключение между режимами отображения."""
        if self.is_point_mode:
            self.point_actor.VisibilityOff()
            self.surface_actor.VisibilityOn()
            self.toggle_button.setText("Переключить на облако точек")
        else:
            self.point_actor.VisibilityOn()
            self.surface_actor.VisibilityOff()
            self.toggle_button.setText("Переключить на поверхность")
        self.is_point_mode = not self.is_point_mode
        self.vtk_widget.GetRenderWindow().Render()

    def copy_volume(self, event):
        """Копирование объёма в буфер обмена."""
        QtWidgets.QApplication.clipboard().setText(self.volume_label.text())

    def export_model(self):
        """Экспорт модели."""
        file, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить модель", "", "PLY (*.ply);;XYZ (*.xyz)")
        if file:
            self.builder.export(file)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()