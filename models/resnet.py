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

class ResnetClassification:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetModel().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def load_from_file(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
    
    def classify(self, state_image_origin):
        # Convert state_image_origin (assumed to be a NumPy array) to a PIL image
        image = Image.fromarray(state_image_origin).convert("RGB")
        
        # Transform and prepare the image
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(image)
            _, predicted_label = torch.max(output, 1)
        
        return predicted_label.item()
