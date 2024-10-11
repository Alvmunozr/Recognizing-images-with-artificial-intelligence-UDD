import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
import cvzone
from polym import PolylineManager
import matplotlib.pyplot as plt

video_path = "SmartFlow/IMG_3513.mp4"

# Definir la ruta del video
video_call = video_path

# Inicializar la transmisión de video
stream = CamGear(source=video_call).start()

# Leer el primer fotograma para obtener la resolución del video
frame = stream.read()
height, width, _ = frame.shape

# Cargar nombres de clases COCO
with open("SmartFlow/coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Cargar el modelo YOLO11n preentrenado
model = YOLO("yolo11n.pt")  # Ajustar para usar el modelo YOLO11n

# Crear una instancia de PolylineManager para manejar las áreas de interés
polyline_manager = PolylineManager()

# Configurar la ventana de OpenCV
cv2.namedWindow('RGB')

# Inicializar el escritor de video con la resolución del video de entrada
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Cambié a XVID
out = cv2.VideoWriter('video_renderizado.avi', fourcc, 20.0, (width, height))  # Resolución ajustada

# Mouse Callback: Definir el evento del mouse para capturar puntos en el video
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polyline_manager.add_point((x, y))

# Establecer el callback del mouse en la ventana 'RGB'
cv2.setMouseCallback('RGB', RGB)

# Variables de conteo de cruce
count = 0
registered_ids = set()  # Conjunto para almacenar IDs de objetos ya registrados
polyline_counts = {}  # Diccionario para almacenar los contadores de Cars y Persons por polilínea

# Loop principal para procesar el video
while True:
    # Leer un fotograma del flujo de video
    frame = stream.read()
    count += 1

    # Procesar solo cada tercer fotograma
    if count % 3 != 0:
        continue
    elif frame is None:
        print("Frame vacío recibido, deteniendo...")
        break

    # Realizar la detección y el seguimiento
    results = model.track(source=frame, conf=0.7, iou=0.5, show=False, persist=True, classes=[0, 2])

    # Si hay detecciones en el fotograma actual
    if results[0].boxes is not None:
        # Obtener las coordenadas de las cajas, IDs de clase, IDs de seguimiento y la confianza
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.cpu().tolist() if results[0].boxes.id is not None else []
        confidences = results[0].boxes.conf.cpu().tolist() if results[0].boxes.conf is not None else []

        # Dibujar las cajas y etiquetas en el fotograma
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box

            # Calcular el centro de la caja delimitadora
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Identificar Persons y Cars
            if class_id == 0:  # Person
                color = (0, 255, 0)
                label = 'Person'
            elif class_id == 2:  # Car
                color = (255, 0, 0)
                label = 'Car'
            else:
                continue

            # Dibujar la caja y la etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'{label}:', (x1, y1 - 10), 1, 1, colorR=color, colorT=(255, 255, 255))

            # Verificar si el objeto cruza alguna polilínea y no ha sido registrado previamente
            for polyline_name in polyline_manager.get_polyline_names():
                if track_id not in registered_ids:
                    if 'car' in polyline_name.lower() and class_id == 2 and polyline_manager.point_polygon_test((cx, cy), polyline_name):
                        if polyline_name not in polyline_counts:
                            polyline_counts[polyline_name] = {'Car': 0, 'Person': 0}
                        polyline_counts[polyline_name]['Car'] += 1
                        registered_ids.add(track_id)
                    elif 'person' in polyline_name.lower() and class_id == 0 and polyline_manager.point_polygon_test((cx, cy), polyline_name):
                        if polyline_name not in polyline_counts:
                            polyline_counts[polyline_name] = {'Car': 0, 'Person': 0}
                        polyline_counts[polyline_name]['Person'] += 1
                        registered_ids.add(track_id)

    # Mostrar los contadores de las polilíneas
    y_offset = 60
    for polyline_name, counts in polyline_counts.items():
        cvzone.putTextRect(frame, f'{polyline_name}: Cars: {counts["Car"]}, Persons: {counts["Person"]}', (50, y_offset), 1, 1)
        y_offset += 30

    # Dibujar las polilíneas y puntos en el fotograma
    frame = polyline_manager.draw_polylines(frame)

    # Guardar el fotograma procesado en el video
    out.write(frame)

    # Mostrar el fotograma procesado
    cv2.imshow("RGB", frame)

    # Manejar eventos de teclado para gestionar las áreas de interés
    if not polyline_manager.handle_key_events():
        break

# Liberar el video y cerrar la ventana
stream.stop()
out.release()
cv2.destroyAllWindows()

# Graficar los datos
Car_counts = [counts["Car"] for counts in polyline_counts.values()]
Person_counts = [counts["Person"] for counts in polyline_counts.values()]

# Crear etiquetas para las polilíneas
labels = list(polyline_counts.keys())

x = np.arange(len(labels))  # Posiciones de las etiquetas
width = 0.35  # Ancho de las barras

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Car_counts, width, label='Cars', color='b')
rects2 = ax.bar(x + width/2, Person_counts, width, label='Persons', color='g')

# Añadir algunas etiquetas
ax.set_xlabel('Polilíneas')
ax.set_ylabel('Cantidad')
ax.set_title('Conteo de Cars y Persons por Polilínea')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

fig.tight_layout()

plt.savefig('bar_graph_1.png')  # Guardar el gráfico antes de mostrarlo
plt.show()
