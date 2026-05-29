# LandUseVision

LandUseVision is a computer vision project that classifies land-use images into four categories:

- Forest
- Industrial
- Permanent Crop
- Residential

The project uses transfer learning with a pretrained ResNet18 model from PyTorch/Torchvision. The final classification layer is adapted for the four project classes, then trained on the included image dataset.

## Project Goal

The goal of this project is to build an image classifier that can recognize different land-use categories from image data. This type of classification is commonly used in remote sensing, urban planning, agriculture, and environmental monitoring.

## Repository Contents

```text
LandUseVision/
  DataSet/
    Train/
    Validate/
    Test/
  results/
  model_utils.py
  train.py
  main.py
  evaluate.py
  predict.py
  demo.py
  model.pth
  requirements.txt
  LICENSE
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

Evaluation and training plots are saved in the `results/` directory by default.

## Setup

This project uses Python with PyTorch and Torchvision. Python 3.10 or newer is recommended.

Clone the repository and move into the project folder:

```bash
git clone https://github.com/IcebergSnow/LandUseVision.git
cd LandUseVision
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

The repository includes a saved model file, `model.pth`, so you can evaluate the model or run predictions without training it first.

Run the demo:

```bash
python demo.py
```

The demo evaluates the saved model, saves `results/confusion_matrix.png`, and predicts the class for one sample image.

Evaluate the saved model on the test set:

```bash
python evaluate.py
```

The evaluation script prints overall accuracy, per-class accuracy, and saves a confusion matrix image to `results/confusion_matrix.png`.

Predict the class for one image:

```bash
python predict.py DataSet/Test/ResidentialTest/Residential_332.jpg
```

The prediction script prints the predicted class and confidence score.

Run the demo with a different image:

```bash
python demo.py --image-path DataSet/Test/ForestTest/Forest_301.jpg
```

Skip evaluation and only run the sample prediction:

```bash
python demo.py --skip-evaluation
```

Train the model again:

```bash
python train.py
```

The training script reports training loss, validation loss, and validation accuracy for each epoch. It saves the best model weights to `model_weights.pth` based on validation accuracy and saves a training curve plot to `results/training_curves.png`.

Custom output paths can be provided with `--confusion-matrix-path` for evaluation and `--metrics-plot-path` for training.

For compatibility, `python main.py` also starts training.

All scripts expect the dataset to be located in the `DataSet/` folder using the structure shown above.

## Checkpoints

New training runs save model weights to `model_weights.pth`, which is the recommended PyTorch checkpoint format for this project.

Evaluation and prediction prefer `model_weights.pth` when it exists. If it does not exist, they fall back to the existing full-model checkpoint, `model.pth`, so the project still works immediately after cloning.

## Device Note

The training, evaluation, and prediction scripts automatically choose between Apple Silicon `mps`, NVIDIA `cuda`, and `cpu` depending on what is available.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
