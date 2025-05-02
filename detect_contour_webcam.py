from ultralytics import YOLO
import cv2
import numpy as np

# Inicializa a webcam
cap = cv2.VideoCapture(0)

# Carrega o modelo YOLOv8 treinado
model = YOLO("runs/segment/train7/weights/best.pt")

# Cor azul claro (BGR)
contour_color = (230, 0, 0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Faz a predição
    results = model(frame)[0]

    if results.masks is not None:
        for mask in results.masks.data:
            # Converte a máscara para uint8
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

            # Encontra contornos
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Desenha os contornos na imagem original
            cv2.drawContours(frame, contours, -1, contour_color, thickness=2)

    # Exibe a imagem com contornos
    cv2.imshow("Contornos de Dentes - YOLOv8", frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
print("Desligando...")
