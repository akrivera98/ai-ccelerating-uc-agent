import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_competiton_solution(output_path: str):
    with open(output_path, "r") as f:
        output_data = json.load(f)

    # Create profiled generation array
    profiled_generation = []
    profiled_units_names = []
    for gen_name, gen_data in output_data["Profiled production (MW)"].items():
        profiled_generation.append(np.array(gen_data))
        profiled_units_names.append(gen_name)

    thermal_generation = []
    thermal_units_names = []
    for gen_name, gen_data in output_data["Thermal production (MW)"].items():
        thermal_generation.append(np.array(gen_data))
        thermal_units_names.append(gen_name)

    charge_rates = []
    storage_units_names = []
    for storage_name, charge_data in output_data["Storage charging rates (MW)"].items():
        charge_rates.append(np.array(charge_data))
        storage_units_names.append(storage_name)

    discharge_rates = []
    for storage_name, discharge_data in output_data[
        "Storage discharging rates (MW)"
    ].items():
        discharge_rates.append(np.array(discharge_data))

    storage_levels = []
    for storage_name, data in output_data["Storage level (MWh)"].items():
        storage_levels.append(np.array(data))

    competition_solution = {
        "profiled_generation": np.stack(profiled_generation),
        "thermal_generation": np.stack(thermal_generation),
        "charge_rate": np.stack(charge_rates),
        "discharge_rate": np.stack(discharge_rates),
        "storage_level": np.stack(storage_levels),
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
