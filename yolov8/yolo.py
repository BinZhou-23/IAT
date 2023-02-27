from ultralytics import YOLO
if __name__ == '__main__':

    # Load a model
    #model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data="e:/yolov8/datasets/test.v1/data.yaml", epochs=1, batch=4,lr0=0.002) #lr0=0.01