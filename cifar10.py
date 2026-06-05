import torch
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset
from typing import Optional, cast

from utils import get_gradcam_mask


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


def plot_sample_with_explanation(sample: torch.Tensor, explanation_mask: torch.Tensor) -> None:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = sample.clone() * std + mean
    image = torch.clamp(image, 0.0, 1.0)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    mask = explanation_mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(mask, cmap="Reds", alpha=0.5, vmin=0.0, vmax=1.0)
    plt.title("Image with Explanation Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_sample_with_explanation(sample: torch.Tensor, explanation_mask: torch.Tensor, attribution_map: Optional[torch.Tensor], filename: str) -> None:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = sample.clone() * std + mean
    image = torch.clamp(image, 0.0, 1.0)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    mask = explanation_mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]

    num_plots = 3 if attribution_map is not None else 2
    plt.figure(figsize=(5 * num_plots, 5))
    
    plt.subplot(1, num_plots, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, num_plots, 2)
    plt.imshow(image)
    plt.imshow(mask, cmap="Reds", alpha=0.5, vmin=0.0, vmax=1.0)
    plt.title("Image with Explanation Mask")
    plt.axis("off")

    if attribution_map is not None:
        plt.subplot(1, num_plots, 3)
        attr_map = attribution_map.detach().cpu().numpy()
        if attr_map.ndim == 3:
            attr_map = attr_map[0]
        plt.imshow(attr_map, cmap="viridis")
        plt.title("Attribution Map")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved sample with explanation to {filename}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset()
    model = load_model(device)
    sample, _ = load_sample(dataset)
    explanation_mask = get_gradcam_mask(model, sample)
    plot_sample_with_explanation(sample, explanation_mask)


if __name__ == "__main__":
    main()
