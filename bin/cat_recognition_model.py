import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from .data import load_known_cats  # 加载猫咪数据库
from .control import activate_relay  # 控制继电器/开关
from .utils import preprocess_image

class CatRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(CatRecognitionModel, self).__init__()
        # 定义模型架构
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # num_classes 需要根据实际猫咪数量设置
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def recognize_cat(image_path):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    known_cats = load_known_cats()
    model = CatRecognitionModel(num_classes=len(known_cats))
    model.load_state_dict(torch.load('cat_recognition_model.pth'))
    model.to(device)
    model.eval()

    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = preprocess_image(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        predictions = model(image_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        max_prob, predicted_idx = torch.max(probabilities, 1)

    # 判断识别结果
    recognized = None
    if max_prob.item() > 0.8:  # 设置置信度阈值
        recognized = known_cats[predicted_idx.item()]

    if recognized:
        return recognized, "已注册猫咪，前端显示"
    else:
        activate_relay()
        return None, "未注册猫咪，触发控制"
