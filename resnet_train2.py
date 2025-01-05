import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import csv
import matplotlib.pyplot as plt

csv.field_size_limit(2147483647)

class GameImageDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.data = []
        self.transform = transform
        self.image_folder = image_folder

        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) 

            for i, row in enumerate(csv_reader):
                if row[0] == "TimeStamp":  
                    continue

                timestamp, action, idx, damage, encoded_image = row
                label = 0 if float(damage) <= 0 else 1
                image_filename = os.path.join(image_folder, f"image_{i}.png")
                if os.path.exists(image_filename):
                    self.data.append((image_filename, label))
                else:
                    print(f"警告: 图像文件 {image_filename} 不存在，跳过此行")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

image_folder = "images"
csv_file = "game_data.csv"
dataset = GameImageDataset(image_folder=image_folder, csv_file=csv_file, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetModel().to(device)

class_counts = [0, 0]
for _, label in dataset:
    class_counts[label] += 1

class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)


def evaluate(loader):
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for label, pred in zip(labels, predicted):
                if label == 0:
                    total_0 += 1
                    if pred == 0:
                        correct_0 += 1
                else:
                    total_1 += 1
                    if pred == 1:
                        correct_1 += 1

    accuracy_0 = 100 * correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = 100 * correct_1 / total_1 if total_1 > 0 else 0
    return accuracy_0, accuracy_1

num_epochs = 10
train_losses = []
test_accuracy_0_list = []
test_accuracy_1_list = []

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    test_accuracy_0, test_accuracy_1 = evaluate(test_loader)
    test_accuracy_0_list.append(test_accuracy_0)
    test_accuracy_1_list.append(test_accuracy_1)
    print(f"测试集上标签为 0 的样本准确率: {test_accuracy_0:.2f}%")
    print(f"测试集上标签为 1 的样本准确率: {test_accuracy_1:.2f}%")

torch.save(model.state_dict(), "resnet_model.pth")
print("模型已保存！")

plt.figure(figsize=(12, 5))

# 绘制 loss 曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracy_0_list, marker='o', label='Accuracy Label 0')
plt.plot(range(1, num_epochs + 1), test_accuracy_1_list, marker='o', label='Accuracy Label 1')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Label 0 and Label 1 Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
