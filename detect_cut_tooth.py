from ultralytics import YOLO
import cv2
import numpy as np
import os

# Caminho de entrada
input_path = input("Envie o caminho da imagem ou pasta: ").strip('"')

# Caminho da imagem de saída
name, _ = os.path.splitext(os.path.basename(input_path))
output_path = os.path.join("images/cut", f"{name}_segment.png")

# Carrega o modelo
model = YOLO("runs/segment/train6/weights/best.pt")

# Função para processar uma imagem
def segment_image(image_path):
    img = cv2.imread(image_path)    
    if img is None:
        print(f"❌ Erro ao carregar a imagem: {image_path}")
        return

    # Faz a predição
    results = model(img)[0]
    mask_final = np.zeros_like(img)

    if results.masks is not None:
        for mask in results.masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_rgb = cv2.merge([mask_np] * 3)
            mask_resized = cv2.resize(mask_rgb, (img.shape[1], img.shape[0]))
            mask_final = cv2.bitwise_or(mask_final, mask_resized)

        resultado = cv2.bitwise_and(img, mask_final)

        # Cria nome de saída
        name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join("images/cut", f"{name}_segment.png")

        cv2.imwrite(output_path, resultado)
        print(f"✅ Segmentado: {output_path}")
    else:
        print(f"⚠️ Nenhuma máscara detectada para: {image_path}")

# Se for uma pasta, processa todas as imagens
if os.path.isdir(input_path):
    print("📂 Pasta detectada. Processando todas as imagens...")
    for file in os.listdir(input_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            segment_image(os.path.join(input_path, file))
else:
    if os.path.isfile(input_path):
        print("🖼️ Imagem única detectada.")
        segment_image(input_path)
    else:
        print("❌ Caminho inválido. Verifique o nome e tente novamente.")
        