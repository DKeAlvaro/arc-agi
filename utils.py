import matplotlib.pyplot as plt
import os
import random

def get_small_sample(challenges, solutions, n=10):
    common_ids = set(challenges.keys()) & set(solutions.keys())
    
    common_ids_list = list(common_ids)
    sampled_ids = random.sample(common_ids_list, n)
    
    sampled_challenges = {challenge_id: challenges[challenge_id] for challenge_id in sampled_ids}
    sampled_solutions = {challenge_id: solutions[challenge_id] for challenge_id in sampled_ids}
    
    return sampled_challenges, sampled_solutions    


def plot_pairs(data, model_solutions=None, true_solutions=None, title="", pdf_pages=None):
    """Plots input-output pairs, with optional model and true solutions."""
    n_pairs = len(data)

    n_cols = 2
    if model_solutions is not None:
        n_cols += 1

    fig, axes = plt.subplots(n_pairs, n_cols, figsize=(2 * n_cols, 2 * n_pairs))

    # Handle single row case
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for i, pair in enumerate(data):
        col_idx = 0

        # Plot input
        axes[i, col_idx].imshow(pair['input'], cmap='viridis')
        axes[i, col_idx].set_title(f'Input {i+1}')
        axes[i, col_idx].axis('off')
        col_idx += 1

        # Plot expected output if available (for training)
        if 'output' in pair:
            axes[i, col_idx].imshow(pair['output'], cmap='viridis')
            axes[i, col_idx].set_title(f'Expected {i+1}')
            axes[i, col_idx].axis('off')
            col_idx += 1
        # Plot true solution if available (for test)
        elif true_solutions is not None and i < len(true_solutions):
            axes[i, col_idx].imshow(true_solutions[i], cmap='viridis')
            axes[i, col_idx].set_title(f'True {i+1}')
            axes[i, col_idx].axis('off')
            col_idx += 1

        # Plot model solution if provided
        if model_solutions is not None and i < len(model_solutions):
            axes[i, col_idx].imshow(model_solutions[i], cmap='viridis')
            axes[i, col_idx].set_title(f'Model {i+1}')
            axes[i, col_idx].axis('off')
            col_idx += 1

    plt.suptitle(title)
    plt.tight_layout()
    if pdf_pages:
        pdf_pages.savefig(fig)
        plt.close(fig)
    else:
        plt.show()