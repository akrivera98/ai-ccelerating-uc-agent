import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_competiton_solution(output_path: str):
    with open(output_path, "r") as f:
        output_data = json.load(f)

    # ---- Profiled generation (sorted by name) ----
    profiled_generation = []
    profiled_units_names = []
    for gen_name in sorted(output_data["Profiled production (MW)"].keys()):
        gen_data = output_data["Profiled production (MW)"][gen_name]
        profiled_generation.append(np.array(gen_data))
        profiled_units_names.append(gen_name)

    # ---- Thermal generation (sorted by name) ----
    thermal_generation = []
    thermal_units_names = []
    for gen_name in sorted(output_data["Thermal production (MW)"].keys()):
        gen_data = output_data["Thermal production (MW)"][gen_name]
        thermal_generation.append(np.array(gen_data))
        thermal_units_names.append(gen_name)

    # ---- Storage charge (sorted by name) ----
    charge_rates = []
    storage_units_names = []
    for storage_name in sorted(output_data["Storage charging rates (MW)"].keys()):
        charge_data = output_data["Storage charging rates (MW)"][storage_name]
        charge_rates.append(np.array(charge_data))
        storage_units_names.append(storage_name)

    # ---- Storage discharge (same sorted order) ----
    discharge_rates = []
    for storage_name in storage_units_names:
        discharge_data = output_data["Storage discharging rates (MW)"][storage_name]
        discharge_rates.append(np.array(discharge_data))

    # ---- Storage levels (same sorted order) ----
    storage_levels = []
    for storage_name in storage_units_names:
        data = output_data["Storage level (MWh)"][storage_name]
        storage_levels.append(np.array(data))

    competition_solution = {
        "profiled_generation": np.stack(profiled_generation).T,
        "thermal_generation": np.stack(thermal_generation).T,
        "charge_rate": np.stack(charge_rates).T,
        "discharge_rate": np.stack(discharge_rates).T,
        "storage_level": np.stack(storage_levels).T,
        "curtailment": np.array(output_data["Load curtail (MW)"]["b1"]),
        "thermal_units_names": thermal_units_names,
        "profiled_units_names": profiled_units_names,
        "storage_units_names": storage_units_names,
    }

    return competition_solution


def load_my_solution(pickle_path: str):
    with open(pickle_path, "rb") as f:
        solution_dict = pickle.load(f)

    non_zero_segprod = np.sum(solution_dict["segprod"], axis=2) > 1e-6

    segprod_sum = np.sum(solution_dict["segprod"], axis=2)
    pmin = np.asarray(solution_dict["pmin"])[:, np.newaxis]
    pmin_matrix = np.where(non_zero_segprod, pmin, 0.0)
    solution_dict["thermal_generation"] = segprod_sum + pmin_matrix
    return solution_dict


def load_my_solution_layer(pickle_path: str):
    with open(pickle_path, "rb") as f:
        solution_dict = pickle.load(f)

    non_zero_segprod = (
        torch.abs(torch.sum(solution_dict["segprod"], dim=-1)) > 1e-1
    )  # changed this tolerance

    segprod_sum = torch.sum(solution_dict["segprod"], dim=-1)
    pmin = torch.asarray(solution_dict["pmin"][:, np.newaxis])
    pmin_matrix = torch.where(non_zero_segprod, pmin, 0.0)
    solution_dict["thermal_generation"] = segprod_sum + pmin_matrix
    return solution_dict


def plot_diff_heatmap(
    var_key: str,
    my_sol: dict,
    ref_sol: dict,
    eps: float = 1e-3,
):
    """
    Percent difference heatmap for any variable stored in dicts.

    pct_diff = 100 * (my - ref) / max(|ref|, eps)

    row_names_map: optional dict mapping var_key -> list of row names
    """
    my_var = (
        my_sol[var_key]
        if isinstance(my_sol[var_key], np.ndarray)
        else my_sol[var_key].squeeze(0).numpy()
    )
    ref_var = ref_sol[var_key]

    assert my_var.shape == ref_var.shape, (
        f"Shape mismatch for {var_key!r}: my {my_var.shape} vs ref {ref_var.shape}"
    )

    N, T = my_var.shape
    if var_key == "thermal_generation":
        row_names = ref_sol["thermal_units_names"]
    elif var_key == "profiled_generation":
        row_names = ref_sol["profiled_units_names"]
    elif var_key in ["charge_rate", "discharge_rate", "storage_level"]:
        row_names = ref_sol["storage_units_names"]

    pct_diff = 100.0 * (my_var - ref_var) / np.maximum(np.abs(ref_var), eps)

    fig, ax = plt.subplots(figsize=(14, 0.35 * N + 2))

    im = ax.imshow(
        pct_diff,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-150,
        vmax=150,
        interpolation="nearest",
    )

    title = f"{var_key} – percent difference"
    ax.set_title(title)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Row")

    ax.set_xticks(range(0, 72))
    ax.set_yticks(range(N))
    ax.set_yticklabels(row_names)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("% difference")

    plt.tight_layout()
    plt.show()


def plot_one_unit(var_key, name, my_sol, ref_sol):
    my_var = (
        my_sol[var_key]
        if isinstance(my_sol[var_key], np.ndarray)
        else my_sol[var_key].squeeze(0).numpy()
    )
    ref_var = ref_sol[var_key]

    if var_key == "thermal_generation":
        row_names = ref_sol["thermal_units_names"]
    elif var_key == "profiled_generation":
        row_names = ref_sol["profiled_units_names"]
    elif var_key in ["charge_rate", "discharge_rate", "storage_level"]:
        row_names = ref_sol["storage_units_names"]
    else:
        raise ValueError(f"Unknown var_key: {var_key}")

    if name not in row_names:
        raise ValueError(f"Unit name {name!r} not found in {var_key} units.")

    idx = row_names.index(name)

    T = my_var.shape[1]
    x = np.arange(T)
    bar_width = 0.4

    plt.figure(figsize=(12, 5))

    plt.bar(
        x - bar_width / 2,
        my_var[idx, :],
        width=bar_width,
        alpha=0.7,
        label="My Solution",
    )

    plt.bar(
        x + bar_width / 2,
        ref_var[idx, :],
        width=bar_width,
        alpha=0.7,
        label="Competition Solution",
    )

    plt.title(f"{var_key.replace('_', ' ').title()} – {name}")
    plt.xlabel("Hour")
    plt.ylabel(var_key)
    plt.xticks([0, 23, 47, 71], ["1", "24", "48", "72"])

    for t in [23.5, 47.5]:
        plt.axvline(t, color="k", linewidth=0.8, alpha=0.5)

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def unpack_lp_solution(y, ed_model_qp):
    """
    y: (nz,) or (1,nz) or (B,nz) from SciPy/HiGHS (numpy) or torch
    ed_model_qp: instance of EDModelQP (or EDModelLP)
    Returns a dict with arrays shaped like competition_solution.
    """
    # sh = ed_model_qp.sh
    # idx = ed_model_qp.idx

    sh = ed_model_qp.form.sh
    idx = ed_model_qp.form.idx

    # to numpy, ensure shape (B, nz)
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None, :]
    B, nz = y.shape
    assert nz == idx.nz, f"Expected nz={idx.nz}, got {nz}"

    prof = np.zeros((B, sh.T, sh.P))
    therm_pa = np.zeros((B, sh.T, sh.G))
    curt = np.zeros((B, sh.T))
    ch = np.zeros((B, sh.T, sh.S))
    dis = np.zeros((B, sh.T, sh.S))
    level = np.zeros((B, sh.T, sh.S))

    for b in range(B):
        for p in range(sh.P):
            for t in range(sh.T):
                prof[b, t, p] = y[b, idx.pg(t, p)]

        for g in range(sh.G):
            for t in range(sh.T):
                therm_pa[b, t, g] = y[b, idx.pa(t, g)]

        for t in range(sh.T):
            curt[b, t] = y[b, idx.curt(t)]

        for s in range(sh.S):
            for t in range(sh.T):
                ch[b, t, s] = y[b, idx.cr(t, s)]
                dis[b, t, s] = y[b, idx.dr(t, s)]
                level[b, t, s] = y[b, idx.s(t, s)]

    out = {
        "profiled_generation": prof.squeeze(0),
        "charge_rate": ch.squeeze(0),
        "discharge_rate": dis.squeeze(0),
        "storage_level": level.squeeze(0),
        "curtailment": curt.squeeze(0),
    }

    # Thermal generation = pmin + pa when pa > 0
    pa = therm_pa.squeeze(0)  # (T,G)
    tol = 1e-6
    nonzero = np.abs(pa) > tol
    pmin = ed_model_qp.form.th_min_power.detach().cpu().numpy()[None, :]
    out["thermal_generation"] = pa + np.where(nonzero, pmin, 0.0)

    out["thermal_units_names"] = ed_model_qp.form.thermal_units_names
    out["profiled_units_names"] = ed_model_qp.form.profiled_units_names
    out["storage_units_names"] = ed_model_qp.form.storage_units_names

    return out
