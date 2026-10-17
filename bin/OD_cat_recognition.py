import wandb
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "resnet34",
    "epochs": 25,
    }
)

# 1. Load the Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_dataset = datasets.ImageFolder(root="/Users/apple/Desktop/research/TNRproject/CatFaceDataset/train", transform=transform)
test_dataset = datasets.ImageFolder(root="/Users/apple/Desktop/research/TNRproject/CatFaceDataset/test", transform=transform)

# 2. Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Define the Model
class CatRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(CatRecognitionModel, self).__init__()
        base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

num_classes = len(train_dataset.classes)  # Number of unique cats
model = CatRecognitionModel(num_classes)

# 4. Initialize Model, Loss, Optimizer, and Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 5. Train the Model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

        # Step the scheduler
        scheduler.step()

        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Train Accuracy: {best_acc:.4f}")

# 6. Extract Features for Comparison (using ResNet34 as feature extractor)
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            outputs = model.features(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten features
            features.append(outputs.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# 7. Test the Model (for classification accuracy)
def test_model(model):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc  # Return the test accuracy

# 8. Predict whether the input image is in the dataset
def predict_image(model, image_path, train_features, train_labels, threshold=0.7):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Get the feature vector of the input image
    with torch.no_grad():
        model.eval()
        feature_vector = model.features(image).view(1, -1).cpu().numpy()

    # Calculate cosine similarity in batches to avoid memory issues
    batch_size = 1000
    n_samples = len(train_features)
    max_similarity = -1
    most_similar_idx = -1
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_similarities = cosine_similarity(feature_vector, train_features[i:end_idx])
        batch_max_idx = np.argmax(batch_similarities)
        batch_max_similarity = batch_similarities[0, batch_max_idx]
        
        if batch_max_similarity > max_similarity:
            max_similarity = batch_max_similarity
            most_similar_idx = i + batch_max_idx
    
    if max_similarity > threshold:
        predicted_label = train_labels[most_similar_idx]
        predicted_class = train_dataset.classes[predicted_label]
        print(f"Input image is most similar to {predicted_class} with similarity score: {max_similarity:.4f}")
    else:
        print("Input image is not similar enough to any image in the dataset.")

# 9. Save and Load the Model
test_accuracy = test_model(model)  # Capture the test accuracy

# Extract features from the training set after training
train_features, train_labels = extract_features(model, train_loader)
# Save the model
save_dict = {
    'model_state_dict': model.state_dict(),
    'epoch': 25,
    'best_accuracy': test_accuracy  # Save the test accuracy
}
torch.save(save_dict, "cat-training.pth")
