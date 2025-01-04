#IMPORTANT: VALIDATION HAS NOT BEEN IMPLEMENTED YET
#maybe add a way to track loss too

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

#Preparing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)), #For ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #ImageNet Normalization Values
])

train_directory = "/Users/iceberg/Downloads/7711810/EuroSAT_RGB/DataSet/Train" #Replace with your own path
validate_directory = "/Users/iceberg/Downloads/7711810/EuroSAT_RGB/DataSet/Validate" #Replace with your own path

train_dataset = datasets.ImageFolder(root=train_directory, transform=transform)
validate_dataset = datasets.ImageFolder(root=validate_directory, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)


#Defining the model (will be using resnet)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  #Pretrained

#CNN stuff
num_features = model.fc.in_features
model.fc= nn.Sequential(
    nn.Linear(num_features, 4),
    nn.Softmax(dim=1)
)

device = torch.device("mps") #Switch to metal
model = model.to(device)
print(device) #To check


#training loop

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

total_epochs = 10
for epoch in range(total_epochs):
  print(f"Epoch:{epoch+1}")
  # Training Loop
  model.train()
  
  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device) #Move the images over to gpu
    
    #Forward Pass
    outputs = model(images)
    loss = loss_function(outputs, labels)
    
    #Backprop 
    optimizer.zero_grad() #Clear gradients
    loss.backward() 
    optimizer.step() #Update parameters
    
   
save_path = "/Users/iceberg/Downloads/7711810/EuroSAT_RGB/model.pth"   #CHANGE PATH TO YOUR OWN
torch.save(model, save_path)
