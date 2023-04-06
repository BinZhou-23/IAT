#https://pysource.com/2023/02/21/yolo-v8-segmentation
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("basket.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        

    def detect(self, img):
        # Get img shape

        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=True)
        result = results[0]
        segmentation_contours_idx = []
        for seg in result.masks.segments:
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores


# Segmentation detector
#ys = YOLOSegmentation("0.935_v8data_640_mosaic64_0.076_0.194_0.0001/weights/best.pt")
ys = YOLOSegmentation("yolov8n-seg.pt")
bboxes, classes, segmentations, scores = ys.detect(img)
for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
    # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
    (x, y, x2, y2) = bbox
    
    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

    cv2.polylines(img, [seg], True, (0, 0, 255), 1)

    cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if class_id == 32:
        print(seg)
        plt.scatter(seg[:,0],seg[:,1])

        # 添加x轴和y轴标签
        plt.xlabel('x')
        plt.ylabel('y')

        # 显示图像
        plt.show()

cv2.imshow("image", img)
cv2.waitKey(0)
