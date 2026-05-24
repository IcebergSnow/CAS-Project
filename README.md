# CAS Project

CAS Project is a computer vision project that classifies land-use images into four categories:

- Forest
- Industrial
- Permanent Crop
- Residential

The project uses transfer learning with a pretrained ResNet18 model from PyTorch/Torchvision. The final classification layer is adapted for the four project classes, then trained on the included image dataset.

## Project Goal

The goal of this project is to build an image classifier that can recognize different land-use categories from image data. This type of classification is commonly used in remote sensing, urban planning, agriculture, and environmental monitoring.

## Repository Contents

```text
CAS-Project/
  DataSet/
    Train/
    Validate/
    Test/
  train.py
  main.py
  evaluate.py
  model.pth
  requirements.txt
```

## Dataset

The dataset is organized into training, validation, and test folders. Each folder contains class-specific subfolders.

```text
DataSet/
  Train/
    Forest/
    Industrial/
    PermanentCrop/
    Residential/
  Validate/
    ForestValidate/
    IndustrialValidate/
    PermanentCropValidate/
    ResidentialValidate/
  Test/
    ForestTest/
    IndustrialTest/
    PermanentCropTest/
    ResidentialTest/
```

The current dataset contains approximately:

| Split | Forest | Industrial | Permanent Crop | Residential |
| --- | ---: | ---: | ---: | ---: |
| Train | 2400 | 2000 | 2000 | 2400 |
| Validate | 300 | 250 | 250 | 300 |
| Test | 300 | 250 | 250 | 300 |

## Model

The training script uses a pretrained `ResNet18` model and replaces the final fully connected layer with a classifier for the four land-use classes.

Image preprocessing includes:

- Resizing images to `224 x 224`
- Converting images to tensors
- Normalizing images with ImageNet mean and standard deviation values

## Setup

This project uses Python with PyTorch and Torchvision. Python 3.10 or newer is recommended.

Clone the repository and move into the project folder:

```bash
git clone https://github.com/IcebergSnow/CAS-Project.git
cd CAS-Project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

The repository already includes a saved model file, `model.pth`, so you can evaluate the model without training it again first.

Evaluate the saved model on the test set:

```bash
python evaluate.py
```

Train the model again:

```bash
python train.py
```

The training script reports training loss, validation loss, and validation accuracy for each epoch. It saves the best model checkpoint to `model.pth` based on validation accuracy.

For compatibility, `python main.py` also starts training.

Both scripts expect the dataset to be located in the `DataSet/` folder using the structure shown above.

## Device Note

The training and evaluation scripts automatically choose between Apple Silicon `mps`, NVIDIA `cuda`, and `cpu` depending on what is available.

## Current Status

The project currently includes:

- A training script using transfer learning
- Validation loss and accuracy tracking during training
- Best-model checkpointing based on validation accuracy
- An evaluation script that reports classification accuracy
- A saved trained model file
- A train/validation/test dataset split
- A requirements file for installing dependencies

## Planned Improvements

Potential next steps include:

- Add training and validation loss graphs
- Add a confusion matrix for test results
- Add a single-image prediction script
- Save model weights separately from model architecture for safer checkpoint loading

## Resume Summary

Built a PyTorch computer vision model using transfer learning with ResNet18 to classify land-use images into forest, industrial, permanent crop, and residential categories. Implemented image preprocessing, model training, and test-set evaluation using a structured train/validation/test dataset.
