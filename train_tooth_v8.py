from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/

def main():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8s-seg.pt")  # load a pretrained model (recommended for training)

    # Use the model
<<<<<<< HEAD:train_among_v8.py
    model.train(data="among.yaml", epochs=200, device=0)  # train the model
=======
    model.train(data="tooth.yaml", epochs=10, device=0)  # train the model
>>>>>>> f016548c43f5f7a10e9e0224789cd557d5667b4b:train_tooth_v8.py
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()
