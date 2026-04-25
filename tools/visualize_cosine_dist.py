import argparse
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_cosine_pairs(features, pids, max_pairs, seed=42):
    features = torch.nn.functional.normalize(features, dim=1, p=2)
    pids = pids.numpy()
    n = len(pids)

    pos_indices = []
    neg_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            if pids[i] == pids[j]:
                pos_indices.append((i, j))
            else:
                neg_indices.append((i, j))

    random.seed(seed)
    if len(pos_indices) > max_pairs:
        pos_indices = random.sample(pos_indices, max_pairs)
    num_neg_sample = len(pos_indices)
    if len(neg_indices) > num_neg_sample:
        neg_indices = random.sample(neg_indices, num_neg_sample)

    def sim_for_pairs(pairs):
        if not pairs:
            return np.array([])
        ii = [p[0] for p in pairs]
        jj = [p[1] for p in pairs]
        fi = features[ii]
        fj = features[jj]
        return (fi * fj).sum(dim=1).numpy()

    return sim_for_pairs(pos_indices), sim_for_pairs(neg_indices)


def load_and_compute(path, max_pairs, split='gallery', seed=42):
    data = torch.load(path, map_location='cpu')
    features = data['{}_features'.format(split)]
    pids = data['{}_pids'.format(split)]
    return compute_cosine_pairs(features, pids, max_pairs, seed)


def plot_single(ax, pos_sim, neg_sim, title):
    bins = np.linspace(-1, 1, 100)
    ax.hist(neg_sim, bins=bins, alpha=0.6, color='royalblue', label='Negative', density=True)
    ax.hist(pos_sim, bins=bins, alpha=0.6, color='tomato', label='Positive', density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="EDITOR Cosine Similarity Distribution")
    parser.add_argument("--features_path", default=None, type=str)
    parser.add_argument("--features_before", default=None, type=str)
    parser.add_argument("--features_after", default=None, type=str)
    parser.add_argument("--output", default="cosine_dist.png", type=str)
    parser.add_argument("--max_pairs", default=50000, type=int)
    parser.add_argument("--split", default="gallery", choices=["query", "gallery"], type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    compare_mode = args.features_before is not None and args.features_after is not None

    if not compare_mode and args.features_path is None:
        parser.error("Provide --features_path or both --features_before and --features_after")

    if compare_mode:
        print('Computing pairs for Before ...')
        pos_b, neg_b = load_and_compute(args.features_before, args.max_pairs, args.split, args.seed)
        print('Computing pairs for After ...')
        pos_a, neg_a = load_and_compute(args.features_after, args.max_pairs, args.split, args.seed)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        plot_single(axes[0], pos_b, neg_b, 'Before')
        plot_single(axes[1], pos_a, neg_a, 'After')
    else:
        print('Computing pairs ...')
        pos_sim, neg_sim = load_and_compute(args.features_path, args.max_pairs, args.split, args.seed)
        print('  Positive pairs: {}, Negative pairs: {}'.format(len(pos_sim), len(neg_sim)))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_single(ax, pos_sim, neg_sim, 'Cosine Similarity Distribution ({})'.format(args.split))

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print('Saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
