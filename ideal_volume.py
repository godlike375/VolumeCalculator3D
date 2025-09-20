import sys
import numpy as np
from scipy import integrate
from typing import List, Tuple, Callable
import pyvista as pv
from pyvistaqt import QtInteractor
import matplotlib.cm as cm
from PyQt6 import QtWidgets, QtGui

# --- Класс ModelBuilder ---
class ModelBuilder:
    def __init__(self, image_width, image_height, scale_x, scale_y):
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = image_width, image_height
        self.scale_x, self.scale_y = scale_x, scale_y
        
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
        
    def _prepare_moments(self, contours: List[np.ndarray], angles: List[float], center: Tuple[float, float] = None):
        center_x = self.IMAGE_WIDTH / 2.0 if center is None else center[0]
        center_y = self.IMAGE_HEIGHT / 2.0 if center is None else center[1]

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
            
            # На вход уже подаются половинные профили
            moments_mm3.append(self._first_moment_of_area(points_mm))
        angles_rad = np.deg2rad(sorted_angles)
        return np.array(moments_mm3), angles_rad

    def volume_radial_integration(self, contours: List[np.ndarray], angles: List[float], center: Tuple[float, float] = None) -> float:
        if len(contours) < 2:
            return 0.0

        # На вход ожидаются половинные профили
        moments_mm3, angles_rad = self._prepare_moments(contours, angles, center)

        use_periodic = False
        if len(angles_rad) >= 3:
            diffs = np.diff(angles_rad)
            h = diffs[0]
            if np.allclose(diffs, h, rtol=1e-7, atol=1e-12):
                L = h * len(angles_rad)
                if np.isclose(L, np.pi, rtol=1e-7, atol=1e-12):
                    use_periodic = True

        if use_periodic:
            h = angles_rad[1] - angles_rad[0]
            volume = h * float(np.sum(moments_mm3))
        else:
            volume = integrate.trapz(y=moments_mm3, x=angles_rad)

        return abs(volume)


# --- Набор тестовых фигур с аналитическими объёмами ---
SHAPE_LIBRARY = {
    "Сфера (точная формула)": {
        "R_base": 5.0,
        "A1": 0.0, "N1": 1, "M1": 1,
        "A2": 0.0, "N2": 1, "M2": 1,
        "volume_formula": lambda p: (4/3) * np.pi * p['R_base']**3
    },
    "Сфера с модуляцией (аналитическая)": {
        "R_base": 5.0,
        "A1": 0.5, "N1": 1, "M1": 0,
        "A2": 0.0, "N2": 1, "M2": 1,
        "volume_formula": lambda p: (4/3)*np.pi*p['R_base']**3 + 4*np.pi*p['R_base']*(p['A1']**2)
    },
    "Эллипсоид вращения": {
        "a": 5.0, "c": 6.0,
        "volume_formula": lambda p: (4/3) * np.pi * p['a']**2 * p['c']
    },
    "Базовая (синус-косинус)": {
        "R_base": 5.0, "A1": 1.5, "N1": 3, "M1": 2,
        "A2": -1.0, "N2": 5, "M2": 4,
    },
    "Звезда (вогнутая)": {
        "R_base": 5.0, "A1": -2.0, "N1": 7, "M1": 3,
        "A2": 1.5, "N2": 9, "M2": 4,
    },
    "Гладкая волна": {
        "R_base": 5.0, "A1": 1.0, "N1": 2, "M1": 1,
        "A2": 0.5, "N2": 4, "M2": 2,
    },
    "Рваная форма": {
        "R_base": 5.0, "A1": -2.5, "N1": 11, "M1": 7,
        "A2": 2.0, "N2": 13, "M2": 9,
    },
    "Сложная вогнутая фигура": {
        "R_base": 5.0, "A1": 3.0, "N1": 5, "M1": 2,
        "A2": -2.5, "N2": 7, "M2": 3,
    },
}

def get_radius(theta: float, phi: float, shape_params: dict) -> float:
    # Для эллипсоида используем специальную параметризацию
    if 'a' in shape_params and 'c' in shape_params:
        # Эллипсоид вращения: r(phi) = sqrt(a^2 sin^2(phi) + c^2 cos^2(phi))
        return np.sqrt(shape_params['a']**2 * np.sin(phi)**2 + shape_params['c']**2 * np.cos(phi)**2)
    
    # Стандартная параметризация для других фигур
    return (shape_params["R_base"] +
            shape_params["A1"] * np.sin(shape_params["N1"] * theta) * np.cos(shape_params["M1"] * phi) +
            shape_params["A2"] * np.cos(shape_params["N2"] * theta) * np.sin(shape_params["M2"] * phi))

def calculate_analytical_volume(shape_params: dict) -> float:
    # Если есть точная формула - используем её
    if 'volume_formula' in shape_params:
        return shape_params['volume_formula'](shape_params)
    
    # Для эллипсоида используем специальную параметризацию
    if 'a' in shape_params and 'c' in shape_params:
        integrand = lambda phi, theta: (1/3) * (get_radius(theta, phi, shape_params)**3) * np.sin(phi)
        volume, _ = integrate.dblquad(integrand, 0, 2 * np.pi, lambda t: 0, lambda t: np.pi)
        return volume
    
    # Стандартный расчёт для других фигур
    integrand = lambda phi, theta: (1/3) * (get_radius(theta, phi, shape_params)**3) * np.sin(phi)
    volume, _ = integrate.dblquad(integrand, 0, 2 * np.pi, lambda t: 0, lambda t: np.pi)
    return volume

def generate_slices(num_slices: int, points_per_contour: int, shape_params: dict):
    contours = []
    angles_deg = np.linspace(0, 180, num_slices, endpoint=False)
    phi_values = np.linspace(0, np.pi, points_per_contour // 2)
    
    for theta_deg in angles_deg:
        theta_rad = np.deg2rad(theta_deg)
        profile_half = []
        
        for phi in phi_values:
            r = get_radius(theta_rad, phi, shape_params)
            profile_half.append([r * np.sin(phi), r * np.cos(phi)])
        
        profile_half = np.array(profile_half)
        profile_mirrored = np.flip(profile_half[1:-1] * [-1, 1], axis=0)
        full_contour = np.vstack([profile_half, profile_mirrored])
        contours.append(full_contour.astype(np.float32))
    
    return contours, list(angles_deg)


# --- GUI приложение ---
class VerificationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Окно верификации алгоритма расчёта объёма")
        self.setGeometry(150, 150, 1000, 800)
        self.plotter = None
        self.current_shape_name = list(SHAPE_LIBRARY.keys())[0]
        self.init_ui()

    def init_ui(self):
        self.create_menu_bar()
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        self.plotter = QtInteractor(central_widget)
        layout.addWidget(self.plotter.interactor)
        self.results_label = QtWidgets.QLabel("Нажмите 'Верификация -> Запустить тест' для начала.")
        self.results_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.results_label)

    def create_menu_bar(self):
        menubar = self.menuBar()
        verification_menu = menubar.addMenu("&Верификация")

        run_action = QtGui.QAction("&Запустить тест", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_verification_test)
        verification_menu.addAction(run_action)

        shape_menu = verification_menu.addMenu("Выбор фигуры")
        for shape_name in SHAPE_LIBRARY.keys():
            act = QtGui.QAction(shape_name, self)
            act.triggered.connect(lambda _, name=shape_name: self.select_shape(name))
            shape_menu.addAction(act)

    def select_shape(self, shape_name: str):
        self.current_shape_name = shape_name
        self.results_label.setText(f"Выбрана фигура: <b>{shape_name}</b>. Нажмите 'Запустить тест'.")

    def run_verification_test(self):
        self.results_label.setText("Идёт расчёт...")
        QtWidgets.QApplication.processEvents()

        IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX = 800, 600
        REAL_WIDTH_MM = 20.0
        SCALE = IMAGE_WIDTH_PX / REAL_WIDTH_MM
        NUM_SLICES_HALF = 405
        POINTS_PER_CONTOUR = 375

        shape_params = SHAPE_LIBRARY[self.current_shape_name]

        # Вычисляем эталонный объём
        ground_truth_volume = calculate_analytical_volume(shape_params)
        
        # Генерируем сечения
        contours_mm, angles = generate_slices(NUM_SLICES_HALF, POINTS_PER_CONTOUR, shape_params)

        center_px = (IMAGE_WIDTH_PX / 2.0, IMAGE_HEIGHT_PX / 2.0)

        # Преобразуем контуры в пиксели
        contours_px = [np.column_stack([
            (c[:, 0] * SCALE) + center_px[0],
            (c[:, 1] * SCALE) + center_px[1] 
        ]).reshape(-1, 1, 2) for c in contours_mm]

        # Разрезаем каждый полный профиль на две половинки и удваиваем список с углами
        full_contours_px = []
        full_angles = []
        for contour_px, ang in zip(contours_px, angles):
            if contour_px is None or len(contour_px) < 3:
                continue
            n_full = contour_px.shape[0]
            n_half = n_full // 2 + 1
            right_half = contour_px[:n_half]
            left_half = contour_px[n_half - 1:]
            full_contours_px.append(right_half)
            full_angles.append(ang)
            full_contours_px.append(left_half)
            full_angles.append((ang + 180.0) % 360.0)

        model_builder = ModelBuilder(IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX, SCALE, SCALE)

        # Вычисляем объём по полному набору половинных профилей (углы 0..360)
        calc_vol_trapezoid = model_builder.volume_radial_integration(full_contours_px, full_angles, center=center_px)

        # Вычисляем ошибки
        err_trap = calc_vol_trapezoid - ground_truth_volume
        rel_err_trap = (err_trap / ground_truth_volume) * 100 if ground_truth_volume != 0 else 0

        # Формируем результат
        result_text = (
            f"<b>Фигура: {self.current_shape_name}</b><br>"
            f"Эталонный объём: {ground_truth_volume:.4f} мм³<br>"
            f"<b>--- Метод Трапеций ---</b><br>"
            f"Рассчитанный объём: {calc_vol_trapezoid:.4f} мм³ | "
            f"<b><font color='{'green' if abs(rel_err_trap) < 0.1 else 'orange'}'>"
            f"Ошибка: {rel_err_trap:+.4f}%</font></b><br>"
        )
        
        # Добавляем информацию о параметрах для сложных фигур
        if "Сложная" in self.current_shape_name or "Звезда" in self.current_shape_name:
            result_text += (
                "<br><b>Примечание:</b> Для этой сложной вогнутой фигуры<br>"
                "аналитический объём вычислен численно с высокой точностью."
            )
        
        self.results_label.setText(result_text)
        self.visualize_model(contours_mm, angles)

    def visualize_model(self, contours_mm: List[np.ndarray], angles_deg: List[float], num_main_slices: int = 18):
        self.plotter.clear()
        full_contours = contours_mm
        full_angles = angles_deg
        all_points_3d = []
        for contour, angle in zip(full_contours, full_angles):
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            points_3d = np.zeros((len(contour), 3))
            points_3d[:, 0] = contour[:, 0] * cos_a
            points_3d[:, 1] = contour[:, 1]
            points_3d[:, 2] = contour[:, 0] * sin_a
            all_points_3d.append(points_3d)
        full_cloud = np.vstack(all_points_3d)
        total_slices = len(full_angles)
        step = max(1, total_slices // num_main_slices)
        rainbow = cm.get_cmap('rainbow', num_main_slices)
        main_slice_count = 0
        for i, points_3d in enumerate(all_points_3d):
            line = pv.lines_from_points(points_3d, close=True)
            if i % step == 0 and main_slice_count < num_main_slices:
                color = rainbow(main_slice_count / num_main_slices)
                self.plotter.add_mesh(line, color=color, line_width=3)
                main_slice_count += 1
            else:
                self.plotter.add_mesh(line, color='#808080', line_width=1.5, opacity=0.5)
        self.plotter.set_background((0.1, 0.1, 0.15))
        self.plotter.add_axes()
        self.plotter.reset_camera()


# --- Точка входа ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VerificationWindow()
    window.show()
    sys.exit(app.exec())