import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import cast
import random
import matplotlib.pyplot as plt


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


def save_sample_with_explanation(sample: torch.Tensor, explanation_mask: torch.Tensor, filename: str) -> None:
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
    plt.savefig(filename)
    print(f"Saved sample with explanation to {filename}")


def get_gradcam_mask(model: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    """Compute a simple Grad-CAM heatmap for a single image sample.

    Returns a single-channel mask with shape (1, H, W) on the CPU in [0,1].
    The function is robust to `sample` having shape (3,H,W) or (1,3,H,W).
    """
    model.eval()

    # prepare input
    if sample.dim() == 3:
        x = sample.unsqueeze(0)
    else:
        x = sample
    device = next(model.parameters()).device
    x = x.to(device)

    # find last convolutional layer in the model
    target_module = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target_module = m

    if target_module is None:
        # fallback: return a zero mask sized like the image
        _, c, h, w = x.shape
        return torch.zeros(1, h, w)

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    fh = target_module.register_forward_hook(forward_hook)
    try:
        bh = target_module.register_full_backward_hook(backward_hook)
    except AttributeError:
        bh = target_module.register_backward_hook(backward_hook)

    # forward pass
    x.requires_grad_(True)
    logits = model(x)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    pred = logits.argmax(dim=1).item()
    score = logits[0, pred]

    model.zero_grad()
    score.backward()

    # remove hooks
    fh.remove()
    bh.remove()

    if activations is None or gradients is None:
        _, c, h, w = x.shape
        return torch.zeros(1, h, w)

    # compute weights: global-average-pool gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    # weighted combination of forward activations
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # upsample to input size
    cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

    cam = cam.squeeze(0)  # (1, H, W)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset()
    model = load_model(device)
    sample, _ = load_sample(dataset)
    explanation_mask = get_gradcam_mask(model, sample)
    plot_sample_with_explanation(sample, explanation_mask)


if __name__ == "__main__":
    main()
