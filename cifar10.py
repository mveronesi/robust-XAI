import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset
from typing import cast
import random


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )


def load_model(device: torch.device) -> nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True,
        trust_repo="check",
    )
    model = cast(nn.Module, model)
    model.to(device)
    model.eval()
    return model


def load_dataset() -> Dataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return torchvision.datasets.CIFAR10(
        root="./cifar10_data",
        train=False,
        download=True,
        transform=transform,
    )


def show_image(image: torch.Tensor) -> None:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = image.clone() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    pil = transforms.ToPILImage()(img)
    # out_path = "sample.png"
    # pil.save(out_path)
    # print(f"Saved sample image to {out_path}")
    pil.show()
    

def load_sample(dataset: Dataset) -> tuple[torch.Tensor, int]:
    """Return a random (image, label) pair from the dataset."""
    idx = random.randrange(len(dataset)) # type: ignore
    image, label = dataset[idx]
    return image, label


def sample_inference(model: nn.Module, dataset: Dataset, device: torch.device) -> None:
    image, label = load_sample(dataset)
    #show_image(image)

    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print(f"Ground truth: {CLASSES[label]}")
    print(f"Prediction:   {CLASSES[pred]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset()
    model = load_model(device)
    sample_inference(model, dataset, device)


if __name__ == "__main__":
    main()
