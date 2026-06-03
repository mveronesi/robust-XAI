import random
import numpy as np
import torch
from torch import nn
from cifar10 import load_model, load_dataset, load_sample, show_image, CLASSES

SEED = 42


def fix_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_under_noise(model: nn.Module, noise_level: float, dimensions_to_perturb: torch.Tensor, sample: torch.Tensor, n_monte_carlo: int, device: torch.device) -> None:
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
    # move tensors to device
    sample = sample.to(device)
    dimensions_to_perturb = dimensions_to_perturb.to(device)
    sample_flat = sample.flatten()
    dimensions_to_perturb_flat = dimensions_to_perturb.flatten()

    # If no noise is requested, avoid constructing a singular covariance matrix
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
    # Map numeric predictions to textual class names and print
    class_names = [CLASSES[int(p)] for p in preds]
    print(f"Predictions on perturbed samples (classes): {class_names}")
    return preds
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = load_dataset()
    sample, label = load_sample(dataset)
    #show_image(sample)
    print(f"Original sample label: {CLASSES[label]}")

    sample_under_noise(
        model=model,
        noise_level=0.01,
        dimensions_to_perturb=torch.ones_like(sample),
        sample=sample,
        n_monte_carlo=10,
        device=device
    )



if __name__ == "__main__":
    fix_random_seeds(SEED)
    main()
    