import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from typing import Sequence
from torch import nn


def plot_radius_trend(radius_trend: Sequence[float], title: str, radius_threshold: float) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(radius_trend)
    plt.title(title)
    plt.xlabel('Number of Perturbed Dimensions')
    plt.ylabel('Certified Radius')
    plt.axhline(y=radius_threshold, color='r', linestyle='--', label=f'Threshold: {radius_threshold}')
    plt.grid()
    plt.legend()
    plt.savefig(f'{title}.png')


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