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
  model_utils.py
  train.py
  main.py
  evaluate.py
  predict.py
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

## Results

The included `model.pth` checkpoint achieves the following results on the test set:

| Metric | Result |
| --- | ---: |
| Overall accuracy | 98.64% |

Per-class accuracy:

| Class | Accuracy | Correct / Total |
| --- | ---: | ---: |
| Forest | 100.00% | 300 / 300 |
| Industrial | 97.60% | 244 / 250 |
| Permanent Crop | 97.20% | 243 / 250 |
| Residential | 99.33% | 298 / 300 |

Example single-image prediction:

```text
Prediction: Residential
Confidence: 100.00%
```

These results can be reproduced with:

```bash
python evaluate.py
```

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

The repository already includes a saved model file, `model.pth`, so you can evaluate the model or run predictions without training it again first.

Evaluate the saved model on the test set:

```bash
python evaluate.py
```

The evaluation script prints overall accuracy, per-class accuracy, and saves a confusion matrix image to `confusion_matrix.png`.

To choose a different confusion matrix output path:

```bash
python evaluate.py --confusion-matrix-path results/confusion_matrix.png
```

Predict the class for one image:

```bash
python predict.py DataSet/Test/ResidentialTest/Residential_332.jpg
```

The prediction script prints the predicted class and confidence score.

Train the model again:

```bash
python train.py
```

The training script reports training loss, validation loss, and validation accuracy for each epoch. It saves the best model weights to `model_weights.pth` based on validation accuracy and saves a training curve plot to `training_curves.png`.

To choose a different training curve output path:

```bash
python train.py --metrics-plot-path results/training_curves.png
```

For compatibility, `python main.py` also starts training.

All scripts expect the dataset to be located in the `DataSet/` folder using the structure shown above.

## Checkpoints

New training runs save model weights to `model_weights.pth`, which is the recommended PyTorch checkpoint format for this project.

Evaluation and prediction prefer `model_weights.pth` when it exists. If it does not exist, they fall back to the existing full-model checkpoint, `model.pth`, so the project still works immediately after cloning.

## Device Note

The training, evaluation, and prediction scripts automatically choose between Apple Silicon `mps`, NVIDIA `cuda`, and `cpu` depending on what is available.

## Current Status

The project currently includes:

- Shared model, transform, device, and checkpoint utilities
- A training script using transfer learning
- Validation loss and accuracy tracking during training
- Best-model weights checkpointing based on validation accuracy
- Training curve generation for loss and validation accuracy
- An evaluation script that reports overall accuracy and per-class accuracy
- Confusion matrix generation for test results
- A single-image prediction script
- A saved trained model file
- A train/validation/test dataset split
- A requirements file for installing dependencies

## Planned Improvements

Potential next steps include:

- Generate and include a `model_weights.pth` checkpoint from a strong training run
- Add saved metric images to the results section
