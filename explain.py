import random
import numpy as np
import torch

from torch import nn
from tqdm import tqdm
from typing import Tuple
from cifar10 import load_model, load_dataset, load_sample, plot_sample_with_explanation, CLASSES
from randomized_smoothing import predict, certify, ABSTAIN

SEED = 42


def fix_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def explain(model: nn.Module, sample: torch.Tensor, radius_threshold: float=0.2) -> Tuple[torch.Tensor, float]:
    # assume feature set is equal to the input dimensions for now
    dimensions_to_perturb = torch.zeros_like(sample)
    # create a randomized cartesian product of (row, col) positions
    H = sample.shape[1]
    W = sample.shape[2]
    flat_indices = torch.randperm(H * W)
    # will iterate over shuffled (row, col) pairs
    row_col_pairs = [(int(idx.item()) // W, int(idx.item()) % W) for idx in flat_indices]
    radius = 0.0
    for row, col in tqdm(row_col_pairs):
        dimensions_to_perturb[:, row, col] = 1.0
        _, radius = certify(
            model=model,
            sample=sample,
            noise_level=0.3,
            dimensions_to_perturb=dimensions_to_perturb,
            n_monte_carlo=100,
            confidence_level=0.95,
            device=sample.device
        )
        #print(f"Certified class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}, certified radius: {radius:.4f}")
        if radius < radius_threshold:
            dimensions_to_perturb[:, row, col] = 0.0
            break
    explanation_mask = 1.0 - dimensions_to_perturb[0]
    print(f"Explanation mask (1 = important, 0 = unimportant):\n{explanation_mask}")
    return explanation_mask, radius


def get_random_explanation_mask(sample: torch.Tensor) -> torch.Tensor:
    mask = (torch.rand_like(sample[0]) > 0.7).float()
    return mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = load_dataset()
    sample, label = load_sample(dataset)
    print(f"Original sample label: {CLASSES[label]}")
    explanation_mask, radius = explain(model, sample, radius_threshold=0.1)
    plot_sample_with_explanation(sample, explanation_mask)
    

if __name__ == "__main__":
    fix_random_seeds(SEED)
    main()
    