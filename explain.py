import random
import numpy as np
import torch
from cifar10 import load_model, load_dataset, load_sample, CLASSES
from randomized_smoothing import predict, certify, ABSTAIN

SEED = 42


def fix_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = load_dataset()
    sample, label = load_sample(dataset)
    print(f"Original sample label: {CLASSES[label]}")
    pred_label = predict(
        model=model,
        sample=sample,
        noise_level=0.04,
        dimensions_to_perturb=torch.ones_like(sample), # for now perturb the whole sample
        n_monte_carlo=1000,
        confidence_level=0.95,
        device=device
    )
    print(f"Predicted class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}")
    # pred_label, radius = certify(
    #     model=model,
    #     sample=sample,
    #     noise_level=0.04,
    #     dimensions_to_perturb=torch.ones_like(sample), # for now perturb the whole sample
    #     n_monte_carlo=1000,
    #     confidence_level=0.95,
    #     device=device
    # )
    # print(f"Certified class: {CLASSES[pred_label] if pred_label != ABSTAIN else 'ABSTAIN'}, certified radius: {radius:.4f}")


if __name__ == "__main__":
    fix_random_seeds(SEED)
    main()
    