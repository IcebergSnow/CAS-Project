import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights



transform = transforms.Compose([
    transforms.Resize((224, 224)), #For ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #ImageNet Normalization Values
])


test_directory = "/Users/iceberg/Downloads/7711810/EuroSAT_RGB/DataSet/Test" #Path to test data
test_dataset = datasets.ImageFolder(root=test_directory, transform=transform) #Same transform as training
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = torch.load("/Users/iceberg/Downloads/7711810/EuroSAT_RGB/model.pth")
model.to(torch.device("mps")) #change if using windows

model.eval()
correct = 0

loss_function = nn.CrossEntropyLoss()

with torch.no_grad(): #dont need to keep track of the gradients
    for images, labels in test_loader:
        images, labels = images.to(torch.device("mps")), labels.to(torch.device("mps"))
        outputs =model(images)
        loss = loss_function(outputs, labels)
        predicted_class =torch.argmax(outputs, dim=1)
        correct +=(predicted_class == labels).sum().item() #Number of correctly predicted images
accuracy = correct / len(test_dataset) * 100 #Calculates the %s
print(f"Accuracy:{accuracy:.2f}%")
