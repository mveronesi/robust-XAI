import torch
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import os
import shutil

from torch import nn
from torch.utils.data import Dataset
from typing import Optional, Sequence, cast

from utils import get_gradcam_mask_custom


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


# TODO: generalize across different datasets
def save_sample_with_explanation(
        sample: torch.Tensor, 
        explanation_mask: torch.Tensor, 
        attribution_map: Optional[torch.Tensor], 
        radius_trend: Sequence[float],
        radius_threshold: float,
        filename: str,
        folder: Optional[str] = None
        ) -> None:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = sample.clone() * std + mean
    image = torch.clamp(image, 0.0, 1.0)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    mask = explanation_mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]

    num_plots = 4 if attribution_map is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("Original Image")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(image)
    ax.set_title("Image with Explanation Mask")
    ax.axis("off")
    ax.imshow(mask, cmap="Reds", alpha=0.5, vmin=0.0, vmax=1.0)

    ax = axes[2]
    ax.plot(range(1, len(radius_trend) + 1), radius_trend)
    ax.axhline(y=radius_threshold, color='r', linestyle='--', label=f'Threshold: {radius_threshold}')
    ax.set_title("Radius Trend")
    ax.set_xlabel("Number of Perturbed Dimensions")
    ax.set_ylabel("Certified Radius")
    ax.grid(True, alpha=0.3)

    if attribution_map is not None:
        ax = axes[3]
        attribution = attribution_map.detach().cpu().numpy()
        if attribution.ndim == 3:
            attribution = attribution[0]
        ax.imshow(attribution, cmap="viridis")
        ax.set_title("Attribution Map")
        ax.axis("off")

    fig.tight_layout()
    if folder:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}/{filename}"
    else:
        filepath = filename
    fig.savefig(filepath)
    print(f"Saved sample with explanation to {filepath}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset()
    model = load_model(device)
    sample, _ = load_sample(dataset)
    explanation_mask = get_gradcam_mask_custom(model, sample)
    plot_sample_with_explanation(sample, explanation_mask)


if __name__ == "__main__":
    main()
