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
  main.py
  TestingCode.py
  model.pth
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

## How to Run

Install the main dependencies:

```bash
pip install torch torchvision tqdm
```

Train the model:

```bash
python main.py
```

Evaluate the saved model on the test set:

```bash
python TestingCode.py
```

## Current Status

The project currently includes:

- A training script using transfer learning
- A test script that reports classification accuracy
- A saved trained model file
- A train/validation/test dataset split

## Planned Improvements

Potential next steps include:

- Use the validation set during training
- Save the best model checkpoint based on validation accuracy
- Add training and validation loss graphs
- Add a confusion matrix for test results
- Add a single-image prediction script
- Improve device selection so the code works on Apple Silicon, CUDA GPUs, and CPUs
- Add a `requirements.txt` file for reproducible setup

## Resume Summary

Built a PyTorch computer vision model using transfer learning with ResNet18 to classify land-use images into forest, industrial, permanent crop, and residential categories. Implemented image preprocessing, model training, and test-set evaluation using a structured train/validation/test dataset.
