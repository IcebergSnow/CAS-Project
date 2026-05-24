import argparse

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

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.2f}%")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), args.weights_path)
            print(f"Saved best model weights to {args.weights_path}")

        print("-" * 50)

    print(f"Best Validation Accuracy: {best_validation_accuracy:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a land-use image classifier.")
    parser.add_argument("--train-dir", default="DataSet/Train")
    parser.add_argument("--validate-dir", default="DataSet/Validate")
    parser.add_argument("--weights-path", default="model_weights.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
