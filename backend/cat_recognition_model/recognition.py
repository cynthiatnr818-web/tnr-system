import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from backend.door_control.controller import send_close_signal


class CatRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(CatRecognitionModel, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CatRecognition:
    def __init__(self, database_dir, model_path, threshold=0.4):
        self.database_dir = database_dir
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

        self.model = CatRecognitionModel(num_classes=2)  
        self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.features(image)
                features = features.view(1, -1).cpu().numpy()
            
            return features
        except Exception as e:
            print(f"error in extracting features for {image_path}: {str(e)}")
            return None

    def extract_features_from_directory(self, directory_path):

        features = []
        image_paths = []
        
        for file in os.listdir(directory_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, file)
                feature = self.extract_features(image_path)
                if feature is not None:
                    features.append(feature)
                    image_paths.append(image_path)
        
        if features:
            features = np.concatenate(features, axis=0)
        return features, image_paths

    def predict_image_custom_database(self, image_path):

        database_features, database_paths = self.extract_features_from_directory(self.database_dir)
        
        if not database_features.size:
            print("no images in database")
            return {'is_match': False, 'max_similarity': 0, 'matched_image': None}
        
        feature_vector = self.extract_features(image_path)
        if feature_vector is None:
            return {'is_match': False, 'max_similarity': 0, 'matched_image': None}
        
        similarities = cosine_similarity(feature_vector, database_features)
        max_similarity = np.max(similarities)
        most_similar_idx = np.argmax(similarities)
        
        result = {
            'is_match': max_similarity > self.threshold,
            'max_similarity': max_similarity,
            'matched_image': database_paths[most_similar_idx] if max_similarity > self.threshold else None
        }
        
        return result
image_paths="/Users/apple/Desktop/TestingDataset"

def check_cat_similarity(image_paths, database_dir="/Users/apple/Desktop/TestingDataset", model_path="/Users/apple/Desktop/research/TNRproject/代码2.0/best_model.pth", threshold=0.5): 
    # Above is the address of the database and model on developer's computer. The data is also attached in the appendix.
    recognizer = CatRecognition(database_dir, model_path, threshold)
    for image_path in image_paths:
        result = recognizer.predict_image_custom_database(image_path)
        if result['is_match']:
            print(f"corresponding images found, similarity: {result['max_similarity']:.2f}")
            send_close_signal()  
            return True
    print(f"no corresponding images, similarity: {result['max_similarity']:.2f}")
    return False

def test_recognition():
    image_path = '/Users/apple/Desktop/TestingInput.jpg' 
#Above is the address of the input image on developer's computer. The data is also attached in the appendix.
    recognizer = CatRecognition(database_dir='/Users/apple/Desktop/TestingDataset', model_path='/Users/apple/Desktop/research/TNRproject/代码2.0/best_model.pth', threshold=0.4)
#Above is the address of the model on developer's computer. The data is also attached in the appendix.
    result = recognizer.predict_image_custom_database(image_path)

    if result['is_match']:
        print(f"匹配成功，相似度: {result['max_similarity']:.2f}, 匹配图片: {result['matched_image']}")
    else:
        print(f"未发现匹配图片，相似度{['max_similarity']}")
image_paths = ["/Users/apple/Desktop/TestingInput.jpg"]  
check_cat_similarity