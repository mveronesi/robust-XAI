import random
import numpy as np
import torch
from torch import nn
from cifar10 import load_model, load_dataset, load_sample, CLASSES
from randomized_smoothing import predict, certify, ABSTAIN

SEED = 42


def fix_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def explain(model: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    # assume feature set is equal to the input dimensions for now
    dimensions_to_perturb = torch.zeros_like(sample)
    # return a random shuffle of the indexes of the sample's elements without considering channels
    row_traversal = torch.randperm(sample.shape[1])
    col_traversal = torch.randperm(sample.shape[2])
    for row in row_traversal:
        for col in col_traversal:
            dimensions_to_perturb[:, row, col] = 1.0
            pred_label, radius = certify(
                model=model,
                sample=sample,
                noise_level=0.1732,
                dimensions_to_perturb=dimensions_to_perturb,
                n_monte_carlo=1000,
                confidence_level=0.95,
                device=sample.device
            )
            print(f"Certified class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}, certified radius: {radius:.4f}")
            if pred_label == ABSTAIN or radius < 0.01:
                dimensions_to_perturb[:, row, col] = 0.0
                break
    explanation_mask = 1.0 - dimensions_to_perturb[0]
    print(f"Explanation mask (1 = important, 0 = unimportant):\n{explanation_mask}")
    return explanation_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = load_dataset()
    sample, label = load_sample(dataset)
    print(f"Original sample label: {CLASSES[label]}")
    # pred_label = predict(
    #     model=model,
    #     sample=sample,
    #     noise_level=0.1732,
    #     dimensions_to_perturb=torch.ones_like(sample), # for now perturb the whole sample
    #     n_monte_carlo=1000,
    #     confidence_level=0.95,
    #     device=device
    # )
    # print(f"Predicted class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}")
    # pred_label, radius = certify(
    #     model=model,
    #     sample=sample,
    #     noise_level=0.1732,
    #     dimensions_to_perturb=torch.ones_like(sample), # for now perturb the whole sample
    #     n_monte_carlo=1000,
    #     confidence_level=0.95,
    #     device=device
    # )
    # print(f"Certified class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}, certified radius: {radius:.4f}")
    explain(model, sample)


if __name__ == "__main__":
    fix_random_seeds(SEED)
    main()
    