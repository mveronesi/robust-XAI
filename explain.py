import random
import numpy as np
import torch
from torch import nn
from typing import Tuple
from scipy.stats import binomtest, beta, norm
from cifar10 import load_model, load_dataset, load_sample, show_image, CLASSES

SEED = 42
ABSTAIN = -1

def fix_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_under_noise(model: nn.Module, sample: torch.Tensor, noise_level: float, dimensions_to_perturb: torch.Tensor, n_monte_carlo: int, device: torch.device) -> torch.Tensor:
    """
    Extract N perturbations, add them to the sample, and run inference on the perturbed samples.
    Args:
        model: The model to run inference with.
        noise_level: The level of noise to add to the sample.
        dimensions_to_perturb: A 0-1 tensor with the same size of input indicating which dimensions to perturb.
        sample: The input sample to perturb and run inference on.
        n_monte_carlo: The number of Monte Carlo samples to generate.
        device: The device to run inference on.
    """
    sample = sample.to(device)
    dimensions_to_perturb = dimensions_to_perturb.to(device)
    sample_flat = sample.flatten()
    dimensions_to_perturb_flat = dimensions_to_perturb.flatten()

    if noise_level == 0 or torch.all(dimensions_to_perturb_flat == 0):
        perturbations = torch.zeros((n_monte_carlo, sample_flat.numel()), device=device)
    else:
        cov_matrix = torch.diag(dimensions_to_perturb_flat * noise_level)
        perturbations = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(sample_flat, device=device),
            covariance_matrix=cov_matrix,
        ).sample((n_monte_carlo,)).to(device)
    perturbed_samples_flat = sample_flat + perturbations
    perturbed_samples = perturbed_samples_flat.reshape(n_monte_carlo, *sample.shape)

    # Show one perturbed sample for inspection
    #show_image(perturbed_samples[0].detach().cpu())
    
    with torch.no_grad():
        logits = model(perturbed_samples)
        preds = logits.argmax(dim=1)
    return preds


# evaluate g at x
def predict(model: nn.Module, sample: torch.Tensor, noise_level: float, dimensions_to_perturb: torch.Tensor, n_monte_carlo: int, confidence_level: float, device: torch.device) -> int:
    preds = sample_under_noise(model, sample, noise_level, dimensions_to_perturb, n_monte_carlo, device)
    counts = torch.bincount(preds, minlength=len(CLASSES))
    top_two = torch.topk(counts, k=2)
    top_two_classes = [CLASSES[int(idx)] for idx in top_two.indices]
    print(
        "Top two predicted classes: "
        f"{top_two_classes[0]} ({int(top_two.values[0])}), "
        f"{top_two_classes[1]} ({int(top_two.values[1])})"
    )
    c_a = top_two.indices[0]
    n_a = top_two.values[0]
    n_b = top_two.values[1]
    p_value = binomtest(n_a, n_a + n_b, p=0.5).pvalue
    print(f"Binomial test p-value: {p_value:.4f} (confidence level: {confidence_level:.2f})")
    if p_value <= 1 - confidence_level:
        return int(c_a)
    else:
        return ABSTAIN
    

# certify the robustness of g around x
def certify(model: nn.Module, sample: torch.Tensor, noise_level: float, dimensions_to_perturb: torch.Tensor, n_monte_carlo: int, confidence_level: float, device: torch.device) -> Tuple[int, float]:
    counts0 = sample_under_noise(model, sample, noise_level, dimensions_to_perturb, 100, device)
    top_class = torch.argmax(torch.bincount(counts0))
    counts = sample_under_noise(model, sample, noise_level, dimensions_to_perturb, n_monte_carlo, device)
    n_a = (counts == top_class).sum().item()
    n_b = (counts != top_class).sum().item()
    alpha = 1 - confidence_level
    p_a_lower = 0.0 if n_a == 0 else float(beta.ppf(alpha, n_a, n_b + 1))
    if p_a_lower > 0.5:
        # radius: noise_level multiplied by the inverse CDF (ppf) of the standard Gaussian at p_a_lower
        radius = noise_level * float(norm.ppf(p_a_lower))
        return int(top_class), radius
    else:
        return ABSTAIN, 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = load_dataset()
    sample, label = load_sample(dataset)
    print(f"Original sample label: {CLASSES[label]}")
    pred_label, radius = certify(
        model=model,
        sample=sample,
        noise_level=0.04,
        dimensions_to_perturb=torch.ones_like(sample), # for now perturb the whole sample
        n_monte_carlo=1000,
        confidence_level=0.95,
        device=device
    )
    print(f"Certified class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}, certified radius: {radius:.4f}")


if __name__ == "__main__":
    fix_random_seeds(SEED)
    main()
    