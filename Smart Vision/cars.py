import torch
import cv2
import time
model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:/Ansagan/Ansagan/yolov10/runs/detect/runs/runs/detect/yolov10x_custom/weights/best.pt", force_reload=True)  # Укажи свой путь к модели
model.conf = 0.5  
video_path = 'cars.mp4'  
cap = cv2.VideoCapture(video_path)
frame_width = 720
frame_height = 640
prev_positions = {}
frame_rate = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30 
pixel_to_meter = 0.05  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))  
    middle_x = frame_width // 2  
    results = model(frame)
    count_forward = 0  
    count_opposite = 0  
    new_positions = {}
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2  
        object_id = f"{int(cls)}_{center_x // 10}_{y1 // 10}"  
        new_positions[object_id] = (center_x, y1)
        speed = 0
        if object_id in prev_positions:
            prev_x, prev_y = prev_positions[object_id]
            distance = ((center_x - prev_x) ** 2 + (y1 - prev_y) ** 2) ** 0.5 * pixel_to_meter
            speed = distance * frame_rate * 3.6  
        if center_x < middle_x:
            count_forward += 1
        else:
            count_opposite += 1 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Class {int(cls)}: {conf:.2f}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Speed: {speed:.1f} km/h', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    prev_positions = new_positions  
    cv2.putText(frame, f'Qarsy: {count_forward}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Bir bagyt: {count_opposite}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.imshow('YOLOv5 Detection', frame)
    time.sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

"""
import torch
import cv2
import time

# Загружаем обученную модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:/Project/yolov5/runs/train/exp/weights/best.pt", force_reload=True)  # Укажи свой путь к модели
model.conf = 0.5  # Уверенность детекции

# Открываем видео
video_path = 'cars.mp4'  # Укажи путь к видео
cap = cv2.VideoCapture(video_path)

# Устанавливаем размер окна видео
frame_width = 800
frame_height = 600

# Словарь для хранения координат машин между кадрами
prev_positions = {}
frame_rate = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Проверка FPS
pixel_to_meter = 0.05  # Примерное соотношение пикселей к метрам, нужно уточнять

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (frame_width, frame_height))  # Изменяем размер кадра
    results = model(frame)
    count = 0  # Счетчик машин на кадре
    new_positions = {}
    
    # Отрисовываем детекции
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        object_id = f"{int(cls)}_{center_x // 10}_{center_y // 10}"  # Улучшенное определение ID
        new_positions[object_id] = (center_x, center_y)
        speed = 0
        
        if object_id in prev_positions:
            prev_x, prev_y = prev_positions[object_id]
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5 * pixel_to_meter
            speed = distance * frame_rate * 3.6  # Переводим в км/ч
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Class {int(cls)}: {conf:.2f}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Speed: {speed:.1f} km/h', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1  # Увеличиваем счетчик машин
    
    prev_positions = new_positions  # Обновляем позиции объектов
    
    # Показываем количество машин внутри видео
    cv2.putText(frame, f'Total Vehicles: {count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Показываем кадр
    cv2.imshow('YOLOv5 Detection', frame)
    time.sleep(0.03)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()




import torch
import cv2
import tkinter as tk
from tkinter import Label
import time

# Инициализация Tkinter
root = tk.Tk()
root.title("YOLOv5 Vehicle Detection")

# Лейбл для подсчета машин
vehicle_count = tk.IntVar()
vehicle_count.set(0)
label = Label(root, textvariable=vehicle_count, font=("Arial", 20))
label.pack()

# Загружаем обученную модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:/Project/yolov5/runs/train/exp/weights/best.pt", force_reload=True)  # Укажи свой путь к модели
model.conf = 0.5  # Уверенность детекции

# Открываем видео
video_path = 'cars.mp4'  # Укажи путь к видео
cap = cv2.VideoCapture(video_path)

# Устанавливаем размер окна видео
frame_width = 800
frame_height = 600

# Словарь для хранения координат машин между кадрами
prev_positions = {}
frame_rate = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Проверка FPS
pixel_to_meter = 0.05  # Примерное соотношение пикселей к метрам, нужно уточнять

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (frame_width, frame_height))  # Изменяем размер кадра
    start_time = time.time()
    results = model(frame)
    count = 0  # Счетчик машин на кадре
    new_positions = {}
    
    # Отрисовываем детекции
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        object_id = f"{int(cls)}_{center_x // 10}_{center_y // 10}"  # Улучшенное определение ID
        new_positions[object_id] = (center_x, center_y)
        speed = 0
        
        if object_id in prev_positions:
            prev_x, prev_y = prev_positions[object_id]
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5 * pixel_to_meter
            speed = distance * frame_rate * 3.6  # Переводим в км/ч
        cv2.putText(frame, f'Total Vehicles: {count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Class {int(cls)}: {conf:.2f}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Speed: {speed:.1f} km/h', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1
    
    prev_positions = new_positions  # Обновляем позиции объектов
    vehicle_count.set(count)  # Обновляем лейбл
    
    # Показываем кадр
    cv2.imshow('YOLOv5 Detection', frame)
    root.update()
    time.sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
root.mainloop()




import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Путь к обученной модели YOLOv5
model_path = 'C:/Project/yolov5/runs/train/exp2/weights/best.pt'

# Загрузка обученной модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Инициализация трекера DeepSORT
tracker = DeepSort(max_age=30)

# Открываем видеофайл
cap = cv2.VideoCapture("cars.mp4")

# Подсчет направлений движения
count_opposite = 0
count_same = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Уменьшаем размер видео
    frame = cv2.resize(frame, (800, 450))
    height, width, _ = frame.shape
    
    # Добавляем линию в середине кадра (сдвигаем немного вправо)
    middle_x = (width // 2) + 50
    cv2.line(frame, (middle_x, 0), (middle_x, height), (255, 255, 255), 2)
    
    # Запуск детекции
    results = model(frame)
    detections = []
    
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.numpy()
        label = int(cls)
        if label == 0:  # Класс cars
            if conf > 0.5:  # Уверенность больше 50%
                detections.append(([x1, y1, x2, y2], conf, label))
    
    # Отслеживание объектов
    tracks = tracker.update_tracks(detections, frame=frame)
    frame_opposite = 0
    frame_same = 0
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        velocity = track.get_avg_velocity()
        
        # Определение направления по положению относительно линии
        direction = "Poputi" if center_x > middle_x else "Oppozite"
        if direction == "Poputi":
            frame_same += 1
        else:
            frame_opposite += 1
        
        # Определение цвета рамки
        color = (0, 255, 0) if direction == "Попутное" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {track_id} - {direction}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Обновляем общий счетчик
    count_opposite += frame_opposite
    count_same += frame_same
    
    # Добавляем текст счетчиков на экран
    cv2.putText(frame, f'Oppozite: {count_opposite}', (frame.shape[1] - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f'Poputi: {count_same}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Показываем кадр
    cv2.imshow('Detection & Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('Детекция и отслеживание завершены.')
"""
