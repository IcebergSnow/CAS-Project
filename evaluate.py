import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_class_name(class_name):
    return class_name.removesuffix("Test")


def save_confusion_matrix(confusion_matrix_values, class_names, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(confusion_matrix_values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion_matrix_values.max() / 2
    for row in range(confusion_matrix_values.shape[0]):
        for column in range(confusion_matrix_values.shape[1]):
            value = confusion_matrix_values[row, column]
            text_color = "white" if value > threshold else "black"
            ax.text(column, row, value, ha="center", va="center", color=text_color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the saved land-use image classifier.")
    parser.add_argument("--test-dir", default="DataSet/Test")
    parser.add_argument("--model-path", default="model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confusion-matrix-path", default="confusion_matrix.png")
    return parser.parse_args()


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Using device: {device}")

    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted_class = torch.argmax(outputs, dim=1).cpu()

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted_class.tolist())

    correct = sum(
        predicted == actual
        for predicted, actual in zip(predicted_labels, true_labels)
    )
    accuracy = correct / len(test_dataset) * 100

    class_names = [format_class_name(class_name) for class_name in test_dataset.classes]
    confusion_matrix_values = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )

    print(f"Accuracy: {accuracy:.2f}%")
    print("Per-class accuracy:")
    for index, class_name in enumerate(class_names):
        total = confusion_matrix_values[index].sum()
        class_correct = confusion_matrix_values[index, index]
        class_accuracy = class_correct / total * 100 if total else 0
        print(f"- {class_name}: {class_accuracy:.2f}% ({class_correct}/{total})")

    save_confusion_matrix(
        confusion_matrix_values,
        class_names,
        args.confusion_matrix_path,
    )
    print(f"Saved confusion matrix to {args.confusion_matrix_path}")


if __name__ == "__main__":
    main()
