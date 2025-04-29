# Treinando YoloV8
Fazendo um treinamento da YoloV8

## Criando Ambiente conda
conda create -n yolov8 python=3.12 -y
conda activate yolov8

## Treinando
### Instalando pytorch com CUDA para uso da GPU
Se você tem uma placa de vídeo compatível com CUDA, primeiro instale o PyTorch com CUDA neste link

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Instale o Yolo
pip install ultralytics


### Treinamento #DEIXEI COM (10 EPOCHS PARA TESTE)

python train_tooth_v8.py

## Ao terminar o treinamento
Agora voce pode testar os resultados

### Testando com WebCam
Para testar com WebCam use o arquivo "detectar_usando_webcam.py"

python detectar_usando_webcam.py

### Testando Capturando Tela
Para testar capturando a tela use o arquivo "detectar_capturando_tela.py"

python detectar_capturando_tela.py

https://github.com/webertrevisan
