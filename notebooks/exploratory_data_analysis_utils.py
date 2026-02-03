from src.datasets.simple_dataset import SimpleDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def compute_frequencies(dataset: SimpleDataset, save_path=None):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            commitment_frequencies = pickle.load(f)
        print("Loaded commitment frequencies from", save_path)
        return commitment_frequencies

    is_on_counts = torch.zeros_like(dataset[0]["target"]["is_on"].T)  # T, G
    gen_names = dataset[0]["gen_names"]
    for i in range(len(dataset)):
        targets = dataset[i]["target"]
        is_on = targets["is_on"]  # T, G

        is_on_counts += is_on.T

    commitment_frequencies = {
        "percentage": (is_on_counts / len(dataset)).numpy(),
        "counts": is_on_counts.numpy(),
        "gen_names": gen_names,
    }

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(commitment_frequencies, f)
        print("Saved commitment frequencies to", save_path)

    return commitment_frequencies


def plot_3d_histogram(
    commitment_frequencies,
    plot_counts=False,
    stride_time=1,
    stride_gen=1,
    sort_gens=True,
    sort_by="mean",
):
    frequencies = (
        commitment_frequencies["percentage"]
        if not plot_counts
        else commitment_frequencies["counts"].astype(int)
    )
    T, G = frequencies.shape  # check dims here
    gen_names = commitment_frequencies["gen_names"]

    if sort_gens:
        if sort_by == "mean":
            score = frequencies.mean(axis=0)
        elif sort_by == "max":
            score = frequencies.max(axis=0)
        elif sort_by == "sum":
            score = frequencies.sum(axis=0)
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}")

        order = np.argsort(score)  # ascending order

        frequencies = frequencies[:, order]
        gen_names = [gen_names[i] for i in order]

    t_idx = np.arange(0, T, stride_time)
    g_idx = np.arange(0, G, stride_gen)

    gen_names_strided = [gen_names[i] for i in g_idx]

    TT, GG = np.meshgrid(t_idx, g_idx, indexing="ij")
    xs = TT.ravel()
    ys = GG.ravel()
    zs = np.zeros_like(xs)
    dz = frequencies[TT, GG].ravel()

    dx = 0.5
    dy = 0.5

    fig = plt.figure(figsize=(24, 24))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True)

    ax.set_xlabel("Time (0..T-1)")
    ax.set_zlabel("Commitment frequency")
    ax.set_title("3D commitment frequency volume (time × gen × freq)")

    ax.set_xlim(0, T)
    ax.set_ylim(0, G)
    ax.set_zlim(0, 1.0)
    ax.set_yticks(g_idx + dy / 2)  # center labels on bars
    ax.set_yticklabels(
        gen_names_strided,
        fontsize=7,
        rotation=0,  # <-- parallel to time (x) axis
        ha="left",
        va="center",
    )

    plt.tight_layout()
    plt.show()


def plot_slice_fix_gen(
    gen_name, commitment_frequencies, plot_counts=False
):  # sorting won't be the same as in 3d hist right now.
    # get gen data
    frequencies = (
        commitment_frequencies["percentage"]
        if not plot_counts
        else commitment_frequencies["counts"].astype(int)
    )
    gen_names = commitment_frequencies["gen_names"]
    gen_idx = gen_names.index(gen_name)
    gen_data = frequencies[:, gen_idx]  # T
    T = gen_data.shape[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    t_idx = np.arange(0, T)
    ax.bar(t_idx, gen_data)

    ax.set_xlabel("Time (0..T-1)")
    ax.set_title(f"Generator {gen_name} commitment frequency")

    ax.set_xlim(0, T)

    plt.tight_layout()
    plt.show()


def plot_slice_fix_period(
    t_idx, commitment_frequencies, plot_counts=False
):  # sorting won't be the same as in 3d hist right now.
    frequencies = (
        commitment_frequencies["percentage"]
        if not plot_counts
        else commitment_frequencies["counts"].astype(int)
    )
    # get period data
    period_data = frequencies[t_idx, :]  # G
    G = period_data.shape[0]
    gen_names = commitment_frequencies["gen_names"]
    fig, ax = plt.subplots(figsize=(12, 6))

    g_idx = np.arange(0, G)
    ax.bar(g_idx, period_data)

    ax.set_xlabel("Generator index")
    ax.set_title(f"Period {t_idx} commitment frequency")
    ax.set_xticks(g_idx)
    ax.set_xticklabels(gen_names, rotation=90, fontsize=8)
    ax.set_xlim(0, G)

    plt.tight_layout()
    plt.show()


def plot_heatmap(commitment_frequencies, sort_gens=True, sort_by="mean"):
    frequencies = commitment_frequencies["percentage"]
    counts = commitment_frequencies["counts"].astype(int)
    gen_names = commitment_frequencies["gen_names"]
    T, G = frequencies.shape  # check dims here

    if sort_gens:
        if sort_by == "mean":
            score = frequencies.mean(axis=0)
        elif sort_by == "max":
            score = frequencies.max(axis=0)
        elif sort_by == "sum":
            score = frequencies.sum(axis=0)
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}")

        order = np.argsort(score)  # ascending order

        frequencies = frequencies[:, order]
        counts = counts[:, order]
        gen_names = [gen_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(24, 12))

    im = ax.imshow(
        frequencies,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Commitment frequency")
    
    ax.set_xticks(np.arange(G))
    ax.set_xticklabels(gen_names, rotation=90, fontsize=8)
    ax.set_ylabel("Time")
    ax.set_title("Commitment frequency heatmap (annotated with counts)")

    # -------- annotate cells --------
    for t in range(T):
        for g in range(G):
            val = counts[t, g]
            # choose text color for contrast
            color = "black" if frequencies[t, g] > 0.6 else "white"
            ax.text(
                g,
                t,
                f"{val}",
                ha="center",
                va="center",
                color=color,
            )

    plt.tight_layout()
    plt.show()

    return order  # sorted_index -> original_gen_id
