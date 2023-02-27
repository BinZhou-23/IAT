import torch
import torchvision
from PIL import Image

# 加载训练好的模型
model = torch.load('best-sg.pt')

# 对待分割的图像进行预处理
image = Image.open('datasets/test.v1/valid/images/img_1670504787640_jpg.rf.a621a681395b2953c4d1c0d31f9d134c.jpg') 
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((416, 416)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# 使用模型进行推理
model.eval()
with torch.no_grad():
    output = model(image_tensor)

# 处理实例分割的结果
boxes = []
for i in range(output.shape[0]):
    # 获取每个实例的置信度、类别和边界框信息
    scores = output[i, :, 4]
    class_ids = output[i, :, 5:].argmax(dim=1)
    class_scores = output[i, :, 5:].max(dim=1).values
    xywh = output[i, :, :4]

    # 根据置信度和类别筛选实例
    mask = (scores * class_scores > 0.5)
    scores = scores[mask]
    class_ids = class_ids[mask]
    xywh = xywh[mask]

    # 将边界框坐标转换为左上角和右下角坐标
    xyxy = xywh.new(xywh.shape)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    # 将边界框信息添加到列表中
    for j in range(xyxy.shape[0]):
        boxes.append((class_ids[j].item(), scores[j].item(), xyxy[j, :].tolist()))

# 打印边界框信息
print(boxes)
