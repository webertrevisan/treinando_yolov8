from ultralytics import YOLO
import cv2
import numpy as np
import os

# Loop até o usuário informar um caminho válido de imagem
while True:
    image_path = input('Envie o caminho da imagem: ')

    # Verifica se o caminho é um arquivo existente
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            print("✅ Imagem carregada com sucesso.")
            break
        else:
            print("❌ Arquivo encontrado, mas não é uma imagem válida. Tente novamente.")
    else:
        print("❌ Caminho inválido ou arquivo não encontrado. Tente novamente.")

# Caminho da imagem de saída
output_path = "images/dente_segmentado.png"

# Carrega o modelo
model = YOLO("runs/segment/train/weights/best.pt")

# Carrega a imagem
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

# Faz a predição
results = model(img)[0]

# Cor do contorno (BGR)
contour_color = (0, 255, 0)  # Verde

# Espessura do contorno
thickness = 2

# Aplica segmentação
if results.masks is not None:
    for mask in results.masks.data:
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Redimensiona a máscara para o tamanho da imagem original
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

        # Encontra contornos
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenha os contornos na imagem original
        cv2.drawContours(img, contours, -1, contour_color, thickness)

    # Salva e exibe
    cv2.imwrite(output_path, img)
    print(f"Imagem com contorno salva em: {output_path}")

    cv2.imshow("Contorno dos Dentes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma máscara foi detectada.")