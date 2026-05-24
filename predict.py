import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


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


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the land-use class for one image.")
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument("--model-path", default="model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    print(f"Using device: {device}")

    transform = build_transform()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    predicted_class = CLASS_NAMES[predicted_index.item()]
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
