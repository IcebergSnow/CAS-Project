from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet18_Weights


CLASS_NAMES = [
    "Forest",
    "Industrial",
    "PermanentCrop",
    "Residential",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes=len(CLASS_NAMES), weights=ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_model(weights_path, model_path, device):
    weights_path = Path(weights_path)
    model_path = Path(model_path)

    if weights_path.exists():
        model = build_model(num_classes=len(CLASS_NAMES), weights=None)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model

    model = torch.load(model_path, weights_only=False, map_location=device)
    return model
