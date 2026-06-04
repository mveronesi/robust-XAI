from matplotlib import pyplot as plt
from typing import Sequence


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