import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image

载入预训练的ResNet模型
class ResNetFaceNet(nn.Module):
    def __init__(self):
        super(ResNetFaceNet, self).__init__()
        # 使用ResNet18作为基础模型
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # 替换最后的全连接层为适合特征提取的层（128维特征）
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):
        return self.resnet(x)

# 加载模型
def load_model():
    model = ResNetFaceNet()
    model.eval()  # 设置为评估模式
    return model

# 预处理图像
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # 添加批量维度
    return img

# 提取图像特征
def extract_features(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        feature = model(image)
    return feature.numpy()

加载模型
model = load_model()

# 示例：提取某张猫咪图像的特征
image_path = 'path_to_cat_image.jpg'
cat_feature = extract_features(model, image_path)

# 从数据集中提取所有猫咪图像的特征
def extract_features_from_dataset(dataset_path, model):
    features = {}
    for image_name in os.listdir(dataset_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(dataset_path, image_name)
            feature = extract_features(model, image_path)
            features[image_name] = feature
    return features

# 计算输入图像与数据集中所有猫咪图像的余弦相似度
def find_similar_cat(input_image_path, features, model, threshold=0.7):
    input_feature = extract_features(model, input_image_path)
    max_similarity = -1
    most_similar_cat = None
    
    for cat_name, cat_feature in features.items():
        similarity = cosine_similarity(input_feature, cat_feature)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_cat = cat_name
    
    if max_similarity > threshold:
        print(f"输入图片中的猫咪与数据集中的 '{most_similar_cat}' 最为相似，相似度: {max_similarity}")
    else:
        print(f"输入图片中的猫咪不属于数据集中的任何一只。相似度: {max_similarity}")

# 提取数据集的所有猫咪图像特征
dataset_path = 'path_to_dataset'
cat_features = extract_features_from_dataset(dataset_path, model)

# 输入一张新的猫咪图片进行判断
input_image_path = 'path_to_input_image.jpg'
find_similar_cat(input_image_path, cat_features, model)

# 数据增强操作
def data_augmentation(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

# 示例：进行数据增强
image_path = 'path_to_cat_image.jpg'
img = Image.open(image_path).convert('RGB')
augmented_img = data_augmentation(img)

class CatDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, img_path

# 数据集加载器
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CatDataset(dataset_path='path_to_dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
