import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 定义模型架构（独立于recognition.py）
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

def train_model_with_logging(model, train_loader, criterion, optimizer, scheduler, num_epochs=50):
    """训练模型并记录训练过程"""
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # 保存完整模型状态
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': best_acc,
            }, 'best_model.pth')

        scheduler.step(epoch_loss)
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

    return best_acc

def main():
    # wandb初始化
    wandb.init(
        project="cat-recognition",
        config={
            "learning_rate": 0.001,
            "architecture": "resnet50",
            "epochs": 50,
        }
    )

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root="/Users/apple/Desktop/research/TNRproject/CatFaceDataset",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型和训练组件
    model = CatRecognitionModel(num_classes=len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 训练模型
    best_acc = train_model_with_logging(
        model, train_loader, criterion, optimizer, scheduler
    )
    print(f"Training completed. Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
