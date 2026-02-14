import json
from src.datasets.uc_dataset import UCDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def compute_storage_load_share(dataset, save_path=None, reload=False):
    shares = []

    if save_path is not None and not reload:
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                shares = pickle.load(f)
            print("Loaded storage load shares from", save_path)
            return shares

    for i in range(len(dataset)):
        item = dataset[i]

        load = item["features"]["profiles"][:, 0]  # (T,)
        p_ch = item["charge_rates"]  # (T,S)
        p_dis = item["discharge_rates"]  # (T,S)

        net = (p_dis - p_ch).sum(dim=1)  # (T,)
        share = net / load

        share = torch.clamp(share, min=0.0)

        shares.append(share)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(shares, f)
        print("Saved storage load shares to", save_path)

    return torch.stack(shares, dim=0)  # (N,T)


def plot_share_histogram(shares, bins=40):
    shares = shares.reshape(-1) * 100  # or shares.flatten()
    plt.figure()
    plt.hist(shares, bins=bins)
    plt.xlabel("Storage share of load")
    plt.ylabel("Count")
    plt.title("Distribution of storage load share (all instances & times)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_share_boxplot_vs_time(shares, max_hours=None):
    data = shares.cpu().numpy() * 100  # %

    if max_hours is not None:
        data = data[:, :max_hours]

    plt.figure(figsize=(24, 4))
    plt.boxplot(
        [data[:, t] for t in range(data.shape[1])],
        showfliers=True,
    )
    plt.xlabel("Hour (t)")
    plt.ylabel("% of load supplied by storage")
    plt.title("Distribution of storage contribution vs time")
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()


def compute_frequencies(dataset: UCDataset, save_path=None, reload=False):
    if os.path.exists(save_path) and not reload:
        with open(save_path, "rb") as f:
            commitment_frequencies = pickle.load(f)
        print("Loaded commitment frequencies from", save_path)
        return commitment_frequencies

    is_on_counts = torch.zeros_like(dataset[0]["target"]["is_on"])  # T, G
    charging_counts = torch.zeros_like(dataset[0]["charge_rates"])  # T, S
    discharging_counts = torch.zeros_like(dataset[0]["charge_rates"])  # T, S
    idle_counts = torch.zeros_like(dataset[0]["charge_rates"])  # T, S

    gen_names = dataset[0]["gen_names"]
    storage_names = dataset[0]["storage_names"]

    for i in range(len(dataset)):
        targets = dataset[i]["target"]
        is_on = targets["is_on"]  # T, G
        storage_status = targets["storage_status"]  # T, S, 3

        is_on_counts += is_on
        charging_counts += storage_status[:, :, 0]
        discharging_counts += storage_status[:, :, 1]
        idle_counts += storage_status[:, :, 2]

    commitment_frequencies = {
        "percentage_is_on": (is_on_counts / len(dataset)).numpy(),
        "percentage_charging": (charging_counts / len(dataset)).numpy(),
        "percentage_discharging": (discharging_counts / len(dataset)).numpy(),
        "percentage_idle": (idle_counts / len(dataset)).numpy(),
        "counts": is_on_counts.numpy(),
        "gen_names": gen_names,
        "storage_names": storage_names,
        "charging_counts": charging_counts.numpy(),
        "discharging_counts": discharging_counts.numpy(),
        "idle_counts": idle_counts.numpy(),
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
        commitment_frequencies["percentage_is_on"]
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
        commitment_frequencies["percentage_is_on"]
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
        commitment_frequencies["percentage_is_on"]
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


def plot_heatmap(
    commitment_frequencies,
    unit_type="gens",
    var_type=None,
    sort_units=True,
    sort_by="mean",
):

    if unit_type == "gens":
        frequencies = commitment_frequencies["percentage_is_on"]
        counts = commitment_frequencies["counts"].astype(int)
        unit_names = commitment_frequencies["gen_names"]
    elif unit_type == "storage":
        if var_type is None:
            raise ValueError(
                "var_type must be specified for storage units. Specify one of 'charging', 'discharging', or 'idle'."
            )
        frequencies = commitment_frequencies[f"percentage_{var_type}"]
        counts = commitment_frequencies[f"{var_type}_counts"].astype(int)
        unit_names = commitment_frequencies["storage_names"]

    T, U = frequencies.shape

    if sort_units:
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
        unit_names = [unit_names[i] for i in order]

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
    label_str = (
        f"{var_type} frequency" if var_type is not None else "Commitment frequency"
    )
    cbar.set_label(label_str)

    ax.set_xticks(np.arange(U))
    ax.set_xticklabels(unit_names, rotation=90, fontsize=8)
    ax.set_ylabel("Time")
    title_str = (
        f"{var_type.capitalize()} frequency heatmap"
        if var_type is not None
        else "Commitment frequency heatmap"
    )
    ax.set_title(title_str + " (annotated with counts)", fontsize=14)

    # -------- annotate cells --------
    for t in range(T):
        for u in range(U):
            val = counts[t, u]
            # choose text color for contrast
            color = "black" if frequencies[t, u] > 0.6 else "white"
            ax.text(
                u,
                t,
                f"{val}",
                ha="center",
                va="center",
                color=color,
            )

    plt.tight_layout()
    plt.show()

    return unit_names  # sorted unit names


def compute_expected_commitment_given_load(
    dataset: UCDataset, n_bins=10, max_load=None, save_path=None, reload=False
):
    if save_path is not None and not reload:
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                expected_commitment = pickle.load(f)
            print("Loaded expected commitment from", save_path)
            return expected_commitment

    if max_load is None and reload:
        min_load, max_load, min_net_load, max_net_load = _get_max_load(dataset)

    n_samples = len(dataset)

    # Load bins
    load_bin_edges = np.linspace(min_load, max_load, n_bins + 1)
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
    ax.set_title(
        "Expected number of committed generators given load level", fontsize=14
    )
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
        f"Generator commitment probability given {'net' if net_load else 'gross'} load",
        fontsize=14,
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


def _get_max_load(dataset: UCDataset):
    seen_max_load = float("-inf")
    seen_min_load = float("inf")
    seen_min_net_load = float("inf")
    seen_max_net_load = float("-inf")

    for i in range(len(dataset)):
        load_profile = dataset[i]["features"]["profiles"][:, 0]  # 72,
        wind_profile = dataset[i]["features"]["profiles"][:, 1]
        solar_profile = dataset[i]["features"]["profiles"][:, 2]

        max_load = load_profile.max().item()
        min_load = load_profile.min().item()
        min_net_load = (load_profile - wind_profile - solar_profile).min().item()
        max_net_load = (load_profile - wind_profile - solar_profile).max().item()
        if min_load < seen_min_load:
            seen_min_load = min_load
        if max_load > seen_max_load:
            seen_max_load = max_load
        if min_net_load < seen_min_net_load:
            seen_min_net_load = min_net_load
        if max_net_load > seen_max_net_load:
            seen_max_net_load = max_net_load

    print(
        f"Determined min load: {seen_min_load}, max load: {seen_max_load}, min net load: {seen_min_net_load}, max net load: {seen_max_net_load}"
    )
    return seen_min_load, seen_max_load, seen_min_net_load, seen_max_net_load

def plot_thermal_unit_parameters():
    sample_path = "data/Train_Data/sample_instance.json"
    with open(sample_path, "r") as f:
        instance = json.load(f)

    gen_data = instance["Generators"]
    minuptimes = []
    mindowntimes = []
    costs = []

    