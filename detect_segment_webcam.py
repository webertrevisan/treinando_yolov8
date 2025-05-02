from ultralytics import YOLO
import cv2
import numpy as np

# Inicializa a webcam
cap = cv2.VideoCapture(0)

# Carrega o modelo YOLOv8 treinado para segmentação
model = YOLO("runs/segment/train8/weights/best.pt")

while True:
    success, frame = cap.read()

    if not success:
        break

    # Faz a predição
    results = model(frame)[0]

    # Cria uma imagem preta para desenhar as máscaras
    mask_overlay = np.zeros_like(frame, dtype=np.uint8)

    if results.masks is not None:
        for mask in results.masks.data:
            # Converte a máscara para uint8
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

            # Cria uma imagem com cor azul claro apenas onde a máscara está presente
            blue_light = np.full_like(frame, (230, 0, 0))  # BGR - azul claro
            mask_colored = cv2.bitwise_and(blue_light, blue_light, mask=mask_resized)

            # Adiciona a máscara à sobreposição
            mask_overlay = cv2.add(mask_overlay, mask_colored)

        # Combina a imagem original com a máscara segmentada azul clara
        frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)

    # Mostra o resultado
    cv2.imshow("Segmentação de Dentes (Azul Claro)", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
print("Desligando...")
