import argparse
from pathlib import Path

import torch
from PIL import Image

from model_utils import CLASS_NAMES, build_transform, get_device, load_model


def get_probabilities(outputs):
    if torch.all(outputs >= 0) and torch.allclose(
        outputs.sum(dim=1),
        torch.ones(outputs.size(0), device=outputs.device),
        atol=1e-4,
    ):
        return outputs
    return torch.softmax(outputs, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the land-use class for one image.")
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument("--weights-path", default="model_weights.pth")
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

    model = load_model(args.weights_path, args.model_path, device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = get_probabilities(outputs)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    predicted_class = CLASS_NAMES[predicted_index.item()]
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
