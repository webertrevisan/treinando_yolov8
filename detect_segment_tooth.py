from ultralytics import YOLO
import cv2
import numpy as np

# Caminho da imagem de entrada
image_path = str(input('Envie o caminho da imagem: '))
# Caminho da imagem de saída
output_path = "images/dente_segmentado.png"

# Carrega o modelo
model = YOLO("runs/segment/train6/weights/best.pt")

# Carrega a imagem
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

# Faz a predição
results = model(img)[0]

# Cria máscara vazia com 3 canais
overlay = np.zeros_like(img, dtype=np.uint8)

# Cor para pintar os dentes (BGR) – azul claro, por exemplo
dente_color = (255, 100, 100)  # Você pode trocar por outra cor

# Aplica segmentação
if results.masks is not None:
    for mask in results.masks.data:
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Redimensiona a máscara para o tamanho da imagem original
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

        # Cria máscara colorida para sobrepor
        for c in range(3):
            overlay[:, :, c] += (mask_resized * dente_color[c]).astype(np.uint8)

    # Aplica transparência (alpha blending)
    alpha = 0.5
    resultado = cv2.addWeighted(img, 1.0, overlay, alpha, 0)

    # Salva e exibe
    cv2.imwrite(output_path, resultado)
    print(f"Imagem segmentada salva em: {output_path}")

    cv2.imshow("Segmentação de Dentes", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma máscara foi detectada.")
