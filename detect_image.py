from ultralytics import YOLO
import cv2
import numpy as np

# Caminho da imagem de entrada
image_path = "images/157.jpg"
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

# Cria imagem preta do mesmo tamanho da original
mask_final = np.zeros_like(img)

# Aplica segmentação
if results.masks is not None:
    for mask in results.masks.data:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255

        # Converte para 3 canais
        mask_rgb = cv2.merge([mask_np] * 3)

        # Redimensiona a máscara para o tamanho da imagem original
        mask_rgb_resized = cv2.resize(mask_rgb, (img.shape[1], img.shape[0]))

        # Combina com a imagem final
        mask_final = cv2.bitwise_or(mask_final, mask_rgb_resized)

    # Aplica a máscara segmentada na imagem original
    resultado = cv2.bitwise_and(img, mask_final)

    # Salva a imagem
    cv2.imwrite(output_path, resultado)
    print(f"Imagem segmentada salva em: {output_path}")

    # Mostra a imagem
    cv2.imshow("Segmentação de Dentes", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma máscara foi detectada.")
