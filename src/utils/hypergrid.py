from itertools import combinations_with_replacement, permutations
import numpy as np
import matplotlib.pyplot as plt

from gfn.gym import HyperGrid
from gfn.modules import GFNModule

import torch

def tupleOfSum(n: int, target: int, bound: int):
    for combination in combinations_with_replacement(range(min(bound + 1, target + 1)), n):
        if sum(combination) == target:
            for perm in set(permutations(combination)):
                yield perm

def terminatingProbsFromGfn(pf: GFNModule, env: HyperGrid):
    arrival_probs = {tuple(state.tolist()): 0. for state in env.all_states.tensor}
    terminating_probs = {tuple(state.tolist()): 0. for state in env.all_states.tensor}
    arrival_probs[tuple(env.s0.tolist())] = 1.
    for i in range (env.ndim * (env.height-1)):
        for comb in tupleOfSum(env.ndim, i, env.height-1):
            state = env.states_from_tensor(torch.tensor(list(comb)).unsqueeze(0))
            masks = state.forward_masks
            logits = pf(state)
            logits[~masks] = -float("inf")
            output_probs = torch.softmax(logits, dim=-1)
            terminating_probs[tuple(comb)] = output_probs[0][-1].item() * arrival_probs[comb]
            for j in range(len(output_probs[0])-1):
                if comb[j]<env.height-1:
                    arrival_probs[tuple(comb[i]+1 if i == j else comb[i] for i in range(env.ndim))] += output_probs[0][j].item() * arrival_probs[comb]
    terminating_probs[env.ndim*(env.height-1,)] = arrival_probs[env.ndim*(env.height-1,)]
    return terminating_probs

def l1_error(pf: GFNModule, env: HyperGrid):
    estimated_probs = list(terminatingProbsFromGfn(pf, env).values())
    real_probs = env.true_dist_pmf
    return (np.mean(np.abs(np.array(estimated_probs) - np.array(real_probs)))).item()


def plot_distribution(pf: GFNModule, env: HyperGrid):
    coordinates = env.all_states.tensor
    estimated_probs = list(terminatingProbsFromGfn(pf, env).values())
    real_probs = env.true_dist_pmf

    grid_size = torch.max(coordinates, dim=0).values + 1
    grid_1 = np.zeros(grid_size.tolist())
    grid_2 = np.zeros(grid_size.tolist())

    for coord, intensity1, intensity2 in zip(coordinates, real_probs, estimated_probs):  # type: ignore
        x, y = coord.tolist()
        grid_1[x, y] = intensity1
        grid_2[x, y] = intensity2

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Real distribution
    im1 = axes[0].imshow(grid_1, cmap="viridis", origin="lower")
    axes[0].set_title("Real distribution")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")
    fig.colorbar(im1, ax=axes[0], orientation='vertical')

    # Estimated distribution
    im2 = axes[1].imshow(grid_2, cmap="viridis", origin="lower")
    axes[1].set_title("Estimated distribution")
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_ylabel("Y Coordinate")
    fig.colorbar(im2, ax=axes[1], orientation='vertical')

    plt.tight_layout()
    plt.show()
