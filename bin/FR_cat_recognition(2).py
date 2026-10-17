import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# 定义ResNet-18模型并修改输出层
class ResNetFaceNet(nn.Module):
    def __init__(self):
        super(ResNetFaceNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # 输出128维的特征

    def forward(self, x):
        return self.resnet(x)

# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批次维度
    return image

# 特征提取函数
def extract_features(model, image_path):
    image = preprocess_image(image_path)

    model.eval()
    with torch.no_grad():
        feature = model(image).squeeze(0).cpu().numpy()  # 去掉批次维度并转换为numpy数组
    return feature

# 从训练集提取所有图像的特征
def extract_features_from_dataset(dataset_path, model):
    train_features = {}
    for image_name in os.listdir(dataset_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(dataset_path, image_name)
            feature = extract_features(model, image_path)
            train_features[image_name] = feature
    return train_features

# 聚类：对训练集的图像特征进行K-means聚类
def generate_pseudo_labels(train_features, num_clusters=10):
    feature_matrix = np.array(list(train_features.values()))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    pseudo_labels = {list(train_features.keys())[i]: cluster_labels[i] for i in range(len(cluster_labels))}
    return pseudo_labels

# 评估模型的聚类准确率
def evaluate_clustering_accuracy(test_dataset_path, model, train_features, pseudo_labels, threshold=0.7):
    correct_predictions = 0
    total_predictions = 0
    
    for image_name in os.listdir(test_dataset_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            input_image_path = os.path.join(test_dataset_path, image_name)
            input_feature = extract_features(model, input_image_path)
            
            max_similarity = -1
            predicted_label = None
            
            for cat_name, cat_feature in train_features.items():
                similarity = cosine_similarity([input_feature], [cat_feature])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_label = pseudo_labels[cat_name]
            
            if max_similarity > threshold:
                correct_predictions += 1
            total_predictions += 1

    # Check for zero total predictions to avoid division by zero
    if total_predictions == 0:
        print("No images found in the test dataset.")
        return 0.0  # Return 0.0 accuracy if no images are processed

    accuracy = correct_predictions / total_predictions
    return accuracy

# 主程序
def main():
    # 初始化模型
    model = ResNetFaceNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # 将模型移动到GPU或CPU

    # Add debug print statements
    train_dataset_path = '/Users/apple/Desktop/research/TNRproject/CatFaceDataset/train/DragonLi'
    print(f"Looking for images in: {train_dataset_path}")
    print(f"Files found: {os.listdir(train_dataset_path)}")
    
    train_features = extract_features_from_dataset(train_dataset_path, model)
    print(f"Number of features extracted: {len(train_features)}")

    # 生成伪标签
    pseudo_labels = generate_pseudo_labels(train_features, num_clusters=10)

    # 评估模型的聚类准确率
    test_dataset_path = 'path_to_test_dataset'
    accuracy = evaluate_clustering_accuracy('/Users/apple/Desktop/research/TNRproject/CatFaceDataset/test/DragonLi', model, train_features, pseudo_labels)
    print(f'模型的聚类准确率为: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
