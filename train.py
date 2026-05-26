import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "landusevision-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from model_utils import build_model, build_transform, get_device


def evaluate(model, data_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            running_loss += loss.item()
            predicted_class = torch.argmax(outputs, dim=1)
            correct += (predicted_class == labels).sum().item()
            total += labels.size(0)

    average_loss = running_loss / len(data_loader)
    accuracy = correct / total * 100
    return average_loss, accuracy


def save_training_curves(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(12, 5))

    loss_ax.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    loss_ax.plot(epochs, history["validation_loss"], marker="o", label="Validation Loss")
    loss_ax.set_title("Loss Over Time")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    accuracy_ax.plot(epochs, history["validation_accuracy"], marker="o", color="green")
    accuracy_ax.set_title("Validation Accuracy Over Time")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy (%)")
    accuracy_ax.set_ylim(0, 100)
    accuracy_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(args):
    transform = build_transform()

    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=transform)
    validate_dataset = datasets.ImageFolder(root=args.validate_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Classes: {train_dataset.classes}")

    model = build_model(num_classes=len(train_dataset.classes))
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_accuracy = 0.0
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True,
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            average_batch_loss = running_loss / (batch_idx + 1)

            progress_bar.set_postfix({
                "batch": f"{batch_idx + 1}/{len(train_loader)}",
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{average_batch_loss:.4f}",
            })

        train_loss = running_loss / len(train_loader)
        validation_loss, validation_accuracy = evaluate(
            model,
            validate_loader,
            loss_function,
            device,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.2f}%")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), args.weights_path)
            print(f"Saved best model weights to {args.weights_path}")

        print("-" * 50)

    save_training_curves(history, args.metrics_plot_path)
    print(f"Saved training curves to {args.metrics_plot_path}")
    print(f"Best Validation Accuracy: {best_validation_accuracy:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a land-use image classifier.")
    parser.add_argument("--train-dir", default="DataSet/Train")
    parser.add_argument("--validate-dir", default="DataSet/Validate")
    parser.add_argument("--weights-path", default="model_weights.pth")
    parser.add_argument("--metrics-plot-path", default="training_curves.png")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
