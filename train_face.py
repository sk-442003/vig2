"""Starter face fine-tuning script.
This is a placeholder demonstrating how to fine-tune a torchvision model on a FER-style dataset.
Prepare folders with images organized by label or provide a custom Dataset class.
"""
import argparse
import os
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to image folder with subfolders per class")
    parser.add_argument("--out", default="models/face_resnet.pth")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for X,y in loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={running/len(loader):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'classes': dataset.classes}, args.out)
    print("Saved fine-tuned face model to", args.out)

if __name__ == '__main__':
    main()
