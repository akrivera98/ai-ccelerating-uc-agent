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

    is_on_counts = torch.zeros_like(dataset[0]["target"]["is_on"])  # T, G
    gen_names = dataset[0]["gen_names"]
    for i in range(len(dataset)):
        targets = dataset[i]["target"]
        is_on = targets["is_on"]  # T, G

        is_on_counts += is_on

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

    return gen_names  # sorted gen names


def compute_expected_commitment_given_load(
    dataset: SimpleDataset, n_bins=10, max_load=None, save_path=None, reload=False
):
    if save_path is not None and not reload:
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                expected_commitment = pickle.load(f)
            print("Loaded expected commitment from", save_path)
            return expected_commitment

    if max_load is None and reload:
        max_load, min_net_load, max_net_load = _get_max_load(dataset)

    n_samples = len(dataset)

    # Load bins
    load_bin_edges = np.linspace(0, max_load, n_bins + 1)
    bin_centers = 0.5 * (load_bin_edges[:-1] + load_bin_edges[1:])

    # Net load bins
    net_load_bin_edges = np.linspace(min_net_load, max_net_load, n_bins + 1)
    net_bin_centers = 0.5 * (net_load_bin_edges[:-1] + net_load_bin_edges[1:])

    # Accumulators for gross load
    committed_sum = np.zeros(n_bins, dtype=np.float64)  # sum of commitments in each bin
    counts = np.zeros(n_bins)
    gen_on_sum_gross = np.zeros(
        (n_bins, dataset[0]["target"]["is_on"].shape[1])
    )  # n_bins, G
    bin_gen_counts = np.zeros(n_bins)

    # Accumulators for net load
    committed_sum_net = np.zeros(
        n_bins, dtype=np.float64
    )  # sum of commitments in each bin
    counts_net = np.zeros(n_bins)
    gen_on_sum_net = np.zeros(
        (n_bins, dataset[0]["target"]["is_on"].shape[1])
    )  # n_bins, G
    bin_gen_counts_net = np.zeros(n_bins)

    for i in range(n_samples):
        load_profile = dataset[i]["features"]["profiles"][:, 0].cpu().numpy()  # 72
        wind_profile = dataset[i]["features"]["profiles"][:, 1].cpu().numpy()
        solar_profile = dataset[i]["features"]["profiles"][:, 2].cpu().numpy()
        net_load_profile = load_profile - wind_profile - solar_profile
        is_on = dataset[i]["target"]["is_on"].cpu().numpy()  # T, G
        committed_t = is_on.sum(axis=1)  # T,

        # Asign each t to a bin

        # Gross load binning
        bin_id = np.digitize(load_profile, load_bin_edges, right=False) - 1
        bin_id = np.clip(bin_id, 0, n_bins - 1)

        # Net load binning
        bin_id_net = np.digitize(net_load_profile, net_load_bin_edges, right=False) - 1
        bin_id_net = np.clip(bin_id_net, 0, n_bins - 1)

        # Accumulate commitments for each bin

        # Accumulation for gross load
        np.add.at(committed_sum, bin_id, committed_t)  # sum over committed gens
        np.add.at(counts, bin_id, 1)

        # Accumulation for net load
        np.add.at(committed_sum_net, bin_id_net, committed_t)  # sum over committed gens
        np.add.at(counts_net, bin_id_net, 1)

        for b in range(n_bins):
            mask = bin_id == b
            if mask.any():
                # Gross load gen on accumulation
                gen_on_sum_gross[b] += is_on[mask].sum(axis=0)
                bin_gen_counts[b] += mask.sum()

            mask_net = bin_id_net == b
            if mask_net.any():
                gen_on_sum_net[b] += is_on[mask_net].sum(axis=0)
                bin_gen_counts_net[b] += mask_net.sum()

    # Compute expectation
    expected_gross = committed_sum / np.maximum(counts, 1)
    expected_net = committed_sum_net / np.maximum(counts_net, 1)

    # Gen on probabilities
    gen_on_prob_gross = gen_on_sum_gross / np.maximum(bin_gen_counts[:, None], 1)
    gen_on_prob_net = gen_on_sum_net / np.maximum(bin_gen_counts_net[:, None], 1)

    results = {
        "load_bin_edges_gross": load_bin_edges,
        "load_bin_edges_net": net_load_bin_edges,
        "bin_centers_gross": bin_centers,
        "bin_centers_net": net_bin_centers,
        "expected_commitment_gross": expected_gross,
        "expected_commitment_net": expected_net,
        "counts_gross": counts,
        "counts_net": counts_net,
        "gen_on_prob_gross": gen_on_prob_gross,
        "gen_on_prob_net": gen_on_prob_net,
        "gen_on_sum_gross": gen_on_sum_gross,
        "gen_on_sum_net": gen_on_sum_net,
    }
    # Save results
    if save_path is not None and not reload:
        with open(save_path, "wb") as f:
            pickle.dump(
                results,
                f,
            )
        print("Saved expected commitment to", save_path)

    return results


def plot_expected_commitment_given_load(results):  # Only gross load for now

    load_bin_edges = results["load_bin_edges_gross"]
    bin_centers = results["bin_centers_gross"]
    expected = results["expected_commitment_gross"]
    counts = results["counts_gross"]

    # Plot
    _, ax = plt.subplots(figsize=(12, 6))
    bin_widths = np.diff(load_bin_edges)
    ax.bar(bin_centers, expected, width=bin_widths, align="center")

    # Annotate counts on bars
    for x, y, c in zip(bin_centers, expected, counts):
        ax.text(x, y, str(int(c)), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Load level")
    ax.set_ylabel("Expected number of committed generators")
    ax.set_title("Expected number of committed generators given load level")
    plt.tight_layout()
    plt.show()


def plot_gen_commitment_heatmap(
    result, gen_names, sort=True, annotate=True, net_load=False
):
    """
    x-axis: generators
    y-axis: load bins
    """

    if not net_load:
        gen_on_prob = result["gen_on_prob_gross"]  # (n_bins, G)
        load_bin_edges = result["load_bin_edges_gross"]
        bin_counts = result["counts_gross"]  # (n_bins,)
        gen_on_sum = result["gen_on_sum_gross"]  # (n_bins, G)
    else:
        gen_on_prob = result["gen_on_prob_net"]  # (n_bins, G)
        load_bin_edges = result["load_bin_edges_net"]
        bin_counts = result["counts_net"]  # (n_bins,)
        gen_on_sum = result["gen_on_sum_net"]  # (n_bins, G)

    n_bins, G = gen_on_prob.shape
    assert len(gen_names) == G

    # ---- sort generators (columns) ----
    if sort:
        gen_scores = gen_on_prob.mean(axis=0)  # (G,)
        order = np.argsort(-gen_scores)  # descending
        gen_on_prob = gen_on_prob[:, order]
        gen_names = [gen_names[g] for g in order]
        gen_on_sum = gen_on_sum[:, order]

    # ---- y-axis labels: load bins ----
    bin_centers = 0.5 * (load_bin_edges[:-1] + load_bin_edges[1:])
    ylabels = [f"{bc:.1f}" for bc in bin_centers]

    fig, ax = plt.subplots(figsize=(30, 12))

    im = ax.imshow(
        gen_on_prob,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(generator ON | load bin)")

    # ---- ticks ----
    ax.set_xticks(np.arange(G))
    ax.set_xticklabels(gen_names, rotation=90, fontsize=8)

    ax.set_yticks(np.arange(n_bins))
    ax.set_yticklabels(ylabels, fontsize=8)

    ax.set_xlabel("Generator (sorted by average commitment)")
    ax.set_ylabel("Load level")
    ax.set_title(
        f"Generator commitment probability given {'net' if net_load else 'gross'} load"
    )

    # ---- annotate cells ----
    if annotate:
        for b in range(n_bins):
            for g in range(G):
                p = gen_on_prob[b, g]
                n = int(gen_on_sum[b, g])
                ax.text(
                    g,
                    b,
                    f"{n}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if p > 0.5 else "white",
                )

    plt.tight_layout()
    plt.show()


def _get_max_load(dataset: SimpleDataset):
    seen_max_load = 0.0
    seen_min_net_load = float("inf")
    seen_max_net_load = float("-inf")
    for i in range(len(dataset)):
        load_profile = dataset[i]["features"]["profiles"][:, 0]  # 72,
        wind_profile = dataset[i]["features"]["profiles"][:, 1]
        solar_profile = dataset[i]["features"]["profiles"][:, 2]
        max_load = load_profile.max().item()
        min_net_load = (load_profile - wind_profile - solar_profile).min().item()
        max_net_load = (load_profile - wind_profile - solar_profile).max().item()
        if max_load > seen_max_load:
            seen_max_load = max_load
        if min_net_load < seen_min_net_load:
            seen_min_net_load = min_net_load
        if max_net_load > seen_max_net_load:
            seen_max_net_load = max_net_load

    print(
        f"Determined max load: {seen_max_load}, min net load: {seen_min_net_load}, max net load: {seen_max_net_load}"
    )
    return seen_max_load, seen_min_net_load, seen_max_net_load


# TODO: net load plots

# TODO: storage plots
