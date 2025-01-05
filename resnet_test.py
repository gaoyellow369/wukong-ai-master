import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetModel().to(device)
model.load_state_dict(torch.load("resnet_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

current_dir = os.path.dirname(__file__) 
image_folder = os.path.join(current_dir, "images")  
image_path = os.path.join(image_folder, "image_1.png")  

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  
image = image.to(device)

with torch.no_grad():
    output = model(image)
    _, predicted_label = torch.max(output, 1)

print(f"预测的标签是: {predicted_label.item()}")
