"""Utilities for randomized smoothing style certification.

This module provides simple helpers to evaluate a classifier under additive
Gaussian perturbations restricted to a subset of input dimensions and to
produce a robustness certificate based on Monte Carlo sampling.

Notes
-----
- The functions accept and return PyTorch tensors and run the model on the
  provided `device`.
- The `predict` function performs a binomial test using SciPy; counts are
  converted to Python ints before calling SciPy to avoid type errors.
- If a function cannot make a high-confidence prediction it returns the
  sentinel value defined by `ABSTAIN`.
"""

import torch

from torch import nn
from typing import Tuple
from scipy.stats import binomtest, beta, norm

ABSTAIN = -1


def sample_under_noise(
    model: nn.Module,
    sample: torch.Tensor,
    noise_level: float,
    dimensions_to_perturb: torch.Tensor,
    n_monte_carlo: int,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate model predictions on samples perturbed by Gaussian noise.

    Generates Monte Carlo samples by adding Gaussian noise to the input,
    selectively perturbing only the dimensions specified in
    `dimensions_to_perturb`, then returns the model's predicted classes.

    Parameters
    ----------
    model : nn.Module
        The classifier to evaluate.
    sample : torch.Tensor
        Input sample to perturb.
    noise_level : float
        Standard deviation of the Gaussian noise.
    dimensions_to_perturb : torch.Tensor
        Binary mask indicating which dimensions to perturb (same shape as sample).
    n_monte_carlo : int
        Number of noisy samples to generate.
    device : torch.device
        Device on which to run computations.

    Returns
    -------
    torch.Tensor
        Predicted class indices of shape (n_monte_carlo,).
    """
    sample = sample.to(device)
    dimensions_to_perturb = dimensions_to_perturb.to(device)
    sample_flat = sample.flatten()
    dimensions_to_perturb_flat = dimensions_to_perturb.flatten()

    if noise_level == 0 or torch.all(dimensions_to_perturb_flat == 0):
        perturbations = torch.zeros((n_monte_carlo, sample_flat.numel()), device=device)
    else:
        # Sample Gaussian noise and mask by dimensions_to_perturb
        noise = torch.randn((n_monte_carlo, sample_flat.numel()), device=device)
        perturbations = noise * torch.sqrt(dimensions_to_perturb_flat * (noise_level**2))
    perturbed_samples_flat = sample_flat + perturbations
    perturbed_samples = perturbed_samples_flat.reshape(n_monte_carlo, *sample.shape)

    with torch.no_grad():
        logits = model(perturbed_samples)
        preds = logits.argmax(dim=1)
    return preds



def predict(
    model: nn.Module,
    sample: torch.Tensor,
    noise_level: float,
    dimensions_to_perturb: torch.Tensor,
    n_monte_carlo: int,
    confidence_level: float,
    device: torch.device,
) -> int:
    """Return a high-confidence prediction under randomized perturbations.

    The method performs `n_monte_carlo` noisy evaluations using
    `sample_under_noise` and counts the top predicted classes. It then runs a
    two-sided binomial test (null: equal probability between top two
    classes) to decide whether the most frequent class is statistically
    significantly preferred. If the test passes at the requested
    `confidence_level`, the predicted class index is returned; otherwise the
    function returns `ABSTAIN`.

    Parameters
    ----------
    model, sample, noise_level, dimensions_to_perturb, n_monte_carlo, device:
        See `sample_under_noise`.
    confidence_level:
        Desired confidence (e.g. 0.95). The function returns a prediction only
        if the binomial test rejects the null hypothesis at this confidence.

    Returns
    -------
    int
        Predicted class index, or `ABSTAIN` when the test is inconclusive.
    """
    preds = sample_under_noise(model, sample, noise_level, dimensions_to_perturb, n_monte_carlo, device)
    classes, counts = torch.unique(preds, return_counts=True)
    top_two = torch.topk(counts, k=min(2, counts.numel()))

    # Convert class/count tensors to Python ints for SciPy compatibility
    c_a = int(classes[top_two.indices[0]].item())
    n_a = int(top_two.values[0].item())
    n_b = int(top_two.values[1].item()) if top_two.values.numel() > 1 else 0

    p_value = binomtest(n_a, n_a + n_b, p=0.5).pvalue
    print(f"Binomial test p-value: {p_value:.4f} (confidence level: {confidence_level:.2f})")
    if p_value <= 1 - confidence_level:
        return c_a
    else:
        return ABSTAIN



def certify(
    model: nn.Module,
    sample: torch.Tensor,
    noise_level: float,
    dimensions_to_perturb: torch.Tensor,
    n_monte_carlo: int,
    confidence_level: float,
    device: torch.device,
) -> Tuple[int, float]:
    """Attempt to certify a robustness radius around `sample`.

    Procedure:
    1. Run a small pilot of 100 noisy evaluations to choose the most frequent
       class (``top_class``).
    2. Run `n_monte_carlo` evaluations to estimate the lower bound on the
       class probability using a Beta lower-confidence bound.
    3. If the lower bound on the top class probability exceeds 0.5, convert
       it to a Gaussian radius using the inverse normal CDF. Otherwise return
       `(ABSTAIN, 0.0)`.

    Returns
    -------
    (int, float)
        Tuple containing the certified class index (or `ABSTAIN`) and the
        certified radius (0.0 when no certificate is possible).
    """
    counts0 = sample_under_noise(model, sample, noise_level, dimensions_to_perturb, 100, device)
    top_class = int(torch.argmax(torch.bincount(counts0)).item())
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
