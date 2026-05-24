import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_directory = "DataSet/Test"
test_dataset = datasets.ImageFolder(root=test_directory, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = get_device()
print(f"Using device: {device}")

model = torch.load("model.pth", weights_only=False, map_location=device)
model.to(device)
model.eval()

correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted_class = torch.argmax(outputs, dim=1)
        correct += (predicted_class == labels).sum().item()

accuracy = correct / len(test_dataset) * 100
print(f"Accuracy: {accuracy:.2f}%")
