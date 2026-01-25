# src/utils/profiling.py
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_one_train_step(
    model,
    ed_layer,
    batch: Dict[str, Any],
    criterion,
    device: torch.device,
    ste_round_fn,
    solver_args: Dict[str, Any] | None = None,
    do_backward: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Runs a single training step on ONE batch and returns:
      - loss_value (float)
      - phase_times (dict: phase -> seconds)

    Notes:
      - This does NOT call optimizer.step() by design (keeps runs comparable).
      - ste_round_fn is injected so this util file doesn't need to import your project code.
    """
    phase: Dict[str, float] = {}
    solver_args = solver_args or {}

    # ---- to_device ----
    t0 = time.perf_counter()
    features = {k: v.to(device) for k, v in batch["features"].items()}
    if isinstance(batch["target"], dict):
        targets = {k: v.to(device) for k, v in batch["target"].items()}
    else:
        targets = batch["target"].to(device)
    phase["to_device"] = time.perf_counter() - t0

    # ---- nn_forward ----
    t0 = time.perf_counter()
    outputs_dict = model(features)
    is_charging = outputs_dict["is_charging"]
    is_discharging = outputs_dict["is_discharging"]
    phase["nn_forward"] = time.perf_counter() - t0

    # ---- rounding ----
    t0 = time.perf_counter()
    outputs_dict["is_on_rounded"] = ste_round_fn(outputs_dict["is_on"])
    phase["rounding"] = time.perf_counter() - t0

    # ---- prep ED inputs ----
    t0 = time.perf_counter()
    load = features["profiles"][:, :, 0]
    solar_max = features["profiles"][:, :, 2].unsqueeze(1)
    wind_max = features["profiles"][:, :, 1].unsqueeze(1)
    phase["prep_ed_inputs"] = time.perf_counter() - t0

    # ---- ED solve ----
    t0 = time.perf_counter()
    ed_solution = ed_layer(
        load,
        solar_max,
        wind_max,
        outputs_dict["is_on_rounded"],
        is_charging,
        is_discharging,
        solver_args=solver_args,
    )
    phase["ed_solve"] = time.perf_counter() - t0

    # ---- loss ----
    t0 = time.perf_counter()
    initial_commitment = features["initial_conditions"][:, :, -1] > 0
    initial_status = features["initial_conditions"][:, :, -1]
    loss = criterion(ed_solution, outputs_dict, targets, initial_status, initial_commitment)
    phase["loss"] = time.perf_counter() - t0

    # ---- backward ----
    t0 = time.perf_counter()
    if do_backward:
        model.zero_grad(set_to_none=True)
        loss.backward()
    phase["backward"] = time.perf_counter() - t0

    return float(loss.item()), phase


def benchmark_batchsize_sweep(
    model,
    ed_layer,
    dataset,
    criterion,
    device: torch.device,
    ste_round_fn,
    batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32),
    steps: int = 3,
    warmup: int = 3,
    solver_args: Dict[str, Any] | None = None,
    subset_size: int | None = None,
    drop_last: bool = True,
) -> List[Dict[str, float]]:
    """
    For each batch size:
      - warm up `warmup` steps (not recorded)
      - record `steps` measured steps
      - return summary stats (mean/std/min/max) for total + each phase
      - also returns mean time per sample

    Workers logic intentionally removed (DataLoader uses num_workers=0).
    """
    assert steps >= 1, "steps must be >= 1"
    assert warmup >= 0, "warmup must be >= 0"
    solver_args = solver_args or {}

    # Choose a fixed subset so each batch size sees comparable data
    max_bs = int(max(batch_sizes))
    needed = max_bs * (steps + warmup)

    if subset_size is None:
        subset_size = min(len(dataset), needed)
    else:
        subset_size = min(len(dataset), subset_size)

    subset = Subset(dataset, list(range(subset_size)))

    results: List[Dict[str, float]] = []

    for bs in batch_sizes:
        loader = DataLoader(
            subset,
            batch_size=int(bs),
            shuffle=False,
            drop_last=drop_last,
            pin_memory=(device.type == "cuda"),
        )
        it = iter(loader)

        # ---- warmup ----
        for _ in range(warmup):
            batch = next(it)
            _sync_if_cuda(device)
            run_one_train_step(
                model=model,
                ed_layer=ed_layer,
                batch=batch,
                criterion=criterion,
                device=device,
                ste_round_fn=ste_round_fn,
                solver_args=solver_args,
                do_backward=True,
            )
            _sync_if_cuda(device)

        # ---- measured steps ----
        per_step = defaultdict(list)  # phase -> [times...]
        for _ in range(steps):
            batch = next(it)

            _sync_if_cuda(device)
            t0 = time.perf_counter()

            _, phase = run_one_train_step(
                model=model,
                ed_layer=ed_layer,
                batch=batch,
                criterion=criterion,
                device=device,
                ste_round_fn=ste_round_fn,
                solver_args=solver_args,
                do_backward=True,
            )

            _sync_if_cuda(device)
            total = time.perf_counter() - t0

            per_step["total"].append(total)
            for k, v in phase.items():
                per_step[k].append(v)

        # ---- summarize ----
        summary: Dict[str, float] = {"batch_size": float(bs)}

        for k, vals in per_step.items():
            arr = np.asarray(vals, dtype=np.float64)
            summary[f"{k}_mean"] = float(arr.mean())
            summary[f"{k}_std"] = float(arr.std(ddof=0))
            summary[f"{k}_min"] = float(arr.min())
            summary[f"{k}_max"] = float(arr.max())

        summary["time_per_sample_mean"] = float(summary["total_mean"] / bs)
        results.append(summary)

    return results


def format_sweep_results(results: List[Dict[str, float]]) -> str:
    """
    Pretty console output: shows mean±std for ALL phases found in results.
    Columns are: B, total, then the remaining phases (sorted, with common ones prioritized),
    plus time/sample (mean).
    """
    if not results:
        return "No results."

    # Discover phases from keys like "<phase>_mean"
    phases = set()
    for r in results:
        for k in r.keys():
            if k.endswith("_mean") and k not in ("time_per_sample_mean",):
                phases.add(k[:-5])  # strip "_mean"

    # Prefer a readable order
    preferred = [
        "total",
        "to_device",
        "nn_forward",
        "rounding",
        "prep_ed_inputs",
        "ed_solve",
        "loss",
        "backward",
    ]
    ordered_phases = [p for p in preferred if p in phases] + sorted(phases - set(preferred))

    # Build header
    headers = ["B"] + ordered_phases + ["time/sample"]
    col_w = max(12, max(len(h) for h in headers) + 2)

    def fmt_cell(phase: str, r: Dict[str, float]) -> str:
        mu = r.get(f"{phase}_mean", float("nan"))
        sd = r.get(f"{phase}_std", float("nan"))
        return f"{mu:.4f}±{sd:.4f}"

    lines = []
    lines.append("Batch-size sweep (mean±std), seconds:")
    lines.append(" | ".join(h.ljust(col_w) for h in headers))
    lines.append("-+-".join("-" * col_w for _ in headers))

    for r in results:
        b = int(r["batch_size"])
        row = [str(b).ljust(col_w)]
        for ph in ordered_phases:
            row.append(fmt_cell(ph, r).ljust(col_w))
        row.append(f"{r['time_per_sample_mean']:.4f}".ljust(col_w))
        lines.append(" | ".join(row))

    return "\n".join(lines)
