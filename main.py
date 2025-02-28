#IMPORTANT: VALIDATION HAS NOT BEEN IMPLEMENTED YET
#maybe add a way to track loss too

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

#Preparing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)), #For ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #ImageNet Normalization Values
])

# Set your data and model paths here
train_directory = "DataSet/Train"
validate_directory = "DataSet/Validate"
model_save_path = "model.pth"

train_dataset = datasets.ImageFolder(root=train_directory, transform=transform)
validate_dataset = datasets.ImageFolder(root=validate_directory, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=128, shuffle=False)


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
    model.train()
    running_loss = 0.0
    
    # Create progress bar for each epoch
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{total_epochs}",
        leave=True
    )
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        #Forward Pass
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        #Backprop 
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar description with current loss
        progress_bar.set_postfix({
            'batch': f'{batch_idx+1}/{len(train_loader)}',
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{running_loss/(batch_idx+1):.4f}'
        })
    
    # Print epoch summary
    epoch_loss = running_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Average Loss: {epoch_loss:.4f}")
    print("-" * 50)

torch.save(model, model_save_path)
