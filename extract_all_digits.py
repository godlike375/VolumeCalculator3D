import cv2
import numpy as np
from pathlib import Path

def extract_digit_from_template(template_path, digit_number):
    """Извлекает цифру из шаблона, убирая черные отступы."""
    
    # Загружаем шаблон
    template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        print(f"Не удалось загрузить {template_path}")
        return None
    
    print(f"Обрабатываю шаблон {digit_number}: {template_img.shape}")
    
    # Используем высокий порог бинаризации для выделения только белой цифры
    _, thresh = cv2.threshold(template_img, 220, 255, cv2.THRESH_BINARY)
    
    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ищем самый большой контур (это должна быть цифра)
    if not contours:
        print(f"Контуры не найдены в шаблоне {digit_number}")
        return None
    
    # Сортируем по площади и берем самый большой
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    
    # Получаем bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    # Добавляем небольшой padding
    padding = 1 
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(template_img.shape[1], x + w + padding)
    y2 = min(template_img.shape[0], y + h + padding)
    
    # Вырезаем цифру
    digit_img = template_img[y1:y2, x1:x2]
    
    print(f"  Вырезана цифра {digit_number}: {digit_img.shape}")
    
    # Нормализуем размер
    target_size = (20, 32)  # TARGET_NORM_SIZE
    digit_normalized = cv2.resize(digit_img, target_size, interpolation=cv2.INTER_AREA)
    
    return digit_normalized

def extract_all_digits():
    """Извлекает все цифры 1-9 из их шаблонов."""
    
    templates_dir = Path("templates")
    
    # Обрабатываем цифры 1-9
    for digit in range(1, 10):
        template_path = templates_dir / f"{digit}.png"
        
        if not template_path.exists():
            print(f"Шаблон {digit}.png не найден")
            continue
        
        # Извлекаем цифру
        digit_img = extract_digit_from_template(template_path, digit)
        
        if digit_img is not None:
            # Сохраняем обновленный шаблон
            output_path = templates_dir / f"{digit}.png"
            cv2.imwrite(str(output_path), digit_img)
            print(f"  Сохранен обновленный шаблон: {output_path}")
        else:
            print(f"  Не удалось извлечь цифру {digit}")
    
    print("\nВсе шаблоны обновлены!")

if __name__ == "__main__":
    extract_all_digits() 