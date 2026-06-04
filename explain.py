import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from typing import Tuple, Sequence, Callable
from cifar10 import load_model, load_dataset, load_sample, save_sample_with_explanation, get_gradcam_mask, CLASSES
from randomized_smoothing import certify, ABSTAIN
from utils import plot_radius_trend

SEED = 11


def fix_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_traversal_order(sample: torch.Tensor) -> Sequence[Tuple[int, int]]:
    H = sample.shape[1]
    W = sample.shape[2]
    flat_indices = torch.randperm(H * W)
    row_col_pairs = [(int(idx.item()) // W, int(idx.item()) % W) for idx in flat_indices]
    return row_col_pairs


def gradcam_traversal_order(model: nn.Module, sample: torch.Tensor) -> Sequence[Tuple[int, int]]:
    cam = get_gradcam_mask(model, sample)
    H, W = cam.shape[1], cam.shape[2]
    flat_indices = torch.argsort(cam.view(-1), descending=False)
    row_col_pairs = [(int(idx.item()) // W, int(idx.item()) % W) for idx in flat_indices]
    return row_col_pairs


def explain(
        model: nn.Module, 
        sample: torch.Tensor, 
        traversal_order: Callable[[torch.Tensor], Sequence[Tuple[int, int]]], 
        radius_threshold: float = 0.2
        ) -> Tuple[torch.Tensor, int, float, Sequence[float]]:
    # assume feature set is equal to the input dimensions for now
    H, W = sample.shape[1], sample.shape[2]
    dimensions_to_perturb = torch.zeros((H, W), device=sample.device)
    # will iterate over shuffled (row, col) pairs
    row_col_pairs = traversal_order(sample)
    final_radius = 0.0
    pred_label = ABSTAIN
    radius_trend = []
    for row, col in tqdm(row_col_pairs):
        dimensions_to_perturb[row, col] = 1.0
        new_pred_label, new_radius = certify(
            model=model,
            sample=sample,
            noise_level=0.3,
            dimensions_to_perturb=dimensions_to_perturb.repeat(3, 1, 1),
            n_monte_carlo=100,
            confidence_level=0.95,
            device=sample.device
        )
        radius_trend.append(new_radius)
        if new_radius < radius_threshold:
            dimensions_to_perturb[row, col] = 0.0 # undo the last perturbation since it caused the radius to drop below the threshold
            break
        else:
            final_radius = new_radius
            pred_label = new_pred_label
    explanation_mask = 1.0 - dimensions_to_perturb # invert the mask to indicate which features are important (not perturbed)
    return explanation_mask, pred_label, final_radius, radius_trend


def get_random_explanation_mask(sample: torch.Tensor) -> torch.Tensor:
    mask = (torch.rand_like(sample[0]) > 0.7).float()
    return mask


def main():
    RADIUS_THRESHOLD = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = load_dataset()
    n_samples_to_explain = 5
    for i in range(n_samples_to_explain):
        print(f"\nExplaining sample {i+1}/{n_samples_to_explain}...")
        sample, label = load_sample(dataset)
        print(f"Original sample label: {CLASSES[label]}")
        explanation_mask, certified_label, radius, radius_trend = explain(
            model,
            sample,
            traversal_order=lambda x: gradcam_traversal_order(model, x),
            radius_threshold=RADIUS_THRESHOLD
            )
        print(f"Certified label: {CLASSES[certified_label]} with radius {radius:.3f}")
        save_sample_with_explanation(sample, explanation_mask, filename=f"explanation_{i+1}.png")
        plot_radius_trend(radius_trend, title=f"Radius Trend for Sample {i+1}", radius_threshold=RADIUS_THRESHOLD)
        
    

if __name__ == "__main__":
    fix_random_seeds(SEED)
    main()
    