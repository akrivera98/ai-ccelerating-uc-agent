import time
import torch

# Adjust these imports to your project
from torch.utils.data import DataLoader
from src.ed_models.ed_model_lp_scipy_parallel import EDModelLP
from src.datasets.simple_dataset import SimpleDataset  # or whatever your Dataset class is

def run_objective_and_backward(model, batch_tensors, parallel: bool):
    load, solar, wind, is_on, is_chg, is_dis = batch_tensors

    # fresh graph
    for t in (load, solar, wind, is_on, is_chg, is_dis):
        if t.grad is not None:
            t.grad.zero_()

    model.parallel_solve = parallel

    t0 = time.perf_counter()
    f = model.objective(load, solar, wind, is_on, is_chg, is_dis, highs_time_limit=None)
    loss = f.sum()
    loss.backward()
    t1 = time.perf_counter()

    grads = {
        "load": load.grad.detach().cpu().clone() if load.grad is not None else None,
        "solar": solar.grad.detach().cpu().clone() if solar.grad is not None else None,
        "wind": wind.grad.detach().cpu().clone() if wind.grad is not None else None,
        "is_on": is_on.grad.detach().cpu().clone() if is_on.grad is not None else None,
        "is_chg": is_chg.grad.detach().cpu().clone() if is_chg.grad is not None else None,
        "is_dis": is_dis.grad.detach().cpu().clone() if is_dis.grad is not None else None,
    }
    return f.detach().cpu(), loss.detach().cpu().item(), grads, (t1 - t0)


def max_abs_diff(a, b):
    if a is None or b is None:
        return None
    return torch.max(torch.abs(a - b)).item()


def collate_uc(batch):
    """
    batch: list of dataset items (dicts)
    returns tensors:
      load, solar, wind: (B,T)
      is_on: (B,T,G)
      is_charging, is_discharging: (B,T,S)
    """
    profiles = torch.stack([b["features"]["profiles"] for b in batch], dim=0)  # (B,T,3)
    load = profiles[:, :, 0]
    solar = profiles[:, :, 1]
    wind = profiles[:, :, 2]

    is_on = torch.stack([b["target"]["is_on"] for b in batch], dim=0)  # (B,T,G)

    storage_status = torch.stack(
        [b["target"]["storage_status"] for b in batch], dim=0
    )  # (B,T,S)

    # assumes: 1=charging, 2=discharging
    is_charging = (storage_status == 1).to(is_on.dtype)
    is_discharging = (storage_status == 2).to(is_on.dtype)

    return load, solar, wind, is_on, is_charging, is_discharging


@torch.no_grad()
def run_objective(model, batch_tensors, parallel: bool):
    load, solar, wind, is_on, is_chg, is_dis = batch_tensors

    model.parallel_solve = parallel

    t0 = time.perf_counter()
    f = model.objective(load, solar, wind, is_on, is_chg, is_dis, highs_time_limit=None)
    t1 = time.perf_counter()

    return f.detach().cpu(), (t1 - t0)


def main():
    # ---- user params ----
    data_dir = "data/Train_Data"  # directory containing instance JSON files
    instance_path = "data/Train_Data/instance_2021_Q1_1/InputData.json"  # for EDModelLP/create_data_dict
    B = 128
    lp_workers = 8
    device = "cpu"
    dtype = torch.float64

    print("Loading dataset...")
    ds = SimpleDataset(data_dir=data_dir)

    print("Building one batch...")
    dl = DataLoader(
        ds, batch_size=B, shuffle=False, num_workers=0, collate_fn=collate_uc
    )
    batch_tensors = next(iter(dl))

    # Move to device/dtype
    batch_tensors = tuple(x.to(device=device, dtype=dtype) for x in batch_tensors)

    load, solar, wind, is_on, is_chg, is_dis = batch_tensors
    is_chg = is_chg[:, :, :, 0]
    is_dis = is_dis[:, :, :, 1]
    print("Batch shapes:")
    print(
        f"  load: {tuple(load.shape)}  solar: {tuple(solar.shape)}  wind: {tuple(wind.shape)}"
    )
    print(
        f"  is_on: {tuple(is_on.shape)}  is_chg: {tuple(is_chg.shape)}  is_dis: {tuple(is_dis.shape)}"
    )

    print("\nCreating model...")
    model = EDModelLP(
        instance_path=instance_path,
        device=device,
        dtype=dtype,
        parallel_solve=False,
        lp_workers=lp_workers,
        parallel_min_batch=1,
    )

    # Warm-up (helps avoid one-time overhead dominating your timing)
    print("\nWarm-up run (serial)...")
    _ = model.objective(load, solar, wind, is_on, is_chg, is_dis)

    print("\nRunning SERIAL objective...")
    batch_tensors = (load, solar, wind, is_on, is_chg, is_dis)
    f_ser, t_ser = run_objective(model, batch_tensors, parallel=False)
    print(f"  Serial time:   {t_ser:.4f} sec")

    print("\nRunning PARALLEL objective...")
    f_par, t_par = run_objective(model, batch_tensors, parallel=True)
    print(f"  Parallel time: {t_par:.4f} sec  (lp_workers={lp_workers})")

    max_diff = torch.max(torch.abs(f_ser - f_par)).item()
    print(f"\nMax |f_serial - f_parallel| = {max_diff:.3e}")

    if t_par > 0:
        print(f"Speedup: {t_ser / t_par:.2f}x")

    print("\nFirst 5 objectives (serial):  ", f_ser[:5].numpy())
    print("First 5 objectives (parallel):", f_par[:5].numpy())

        # -----------------------------
    # Backward pass check
    # -----------------------------
    print("\n=== Backward pass check (serial vs parallel) ===")

    # Re-create tensors with grad enabled (important: don't reuse @no_grad ones)
    load = load.detach().clone().requires_grad_(True)
    solar = solar.detach().clone().requires_grad_(True)
    wind = wind.detach().clone().requires_grad_(True)
    is_on = is_on.detach().clone().requires_grad_(True)

    # These are binary indicators; gradient exists in your custom backward even if inputs are 0/1.
    is_chg = is_chg.detach().clone().requires_grad_(True)
    is_dis = is_dis.detach().clone().requires_grad_(True)

    batch_tensors_grad = (load, solar, wind, is_on, is_chg, is_dis)

    # Serial
    f_ser2, loss_ser, g_ser, t_ser_bwd = run_objective_and_backward(
        model, batch_tensors_grad, parallel=False
    )
    print(f"Serial forward+backward time:   {t_ser_bwd:.4f} sec, loss={loss_ser:.6e}")

    # Parallel (need fresh tensors so grads don’t mix)
    load_p = load.detach().clone().requires_grad_(True)
    solar_p = solar.detach().clone().requires_grad_(True)
    wind_p = wind.detach().clone().requires_grad_(True)
    is_on_p = is_on.detach().clone().requires_grad_(True)
    is_chg_p = is_chg.detach().clone().requires_grad_(True)
    is_dis_p = is_dis.detach().clone().requires_grad_(True)

    batch_tensors_grad_p = (load_p, solar_p, wind_p, is_on_p, is_chg_p, is_dis_p)

    f_par2, loss_par, g_par, t_par_bwd = run_objective_and_backward(
        model, batch_tensors_grad_p, parallel=True
    )
    print(f"Parallel forward+backward time: {t_par_bwd:.4f} sec, loss={loss_par:.6e}")

    # Compare losses/outputs
    print(f"Max |f_ser - f_par|: {torch.max(torch.abs(f_ser2 - f_par2)).item():.3e}")
    print(f"|loss_ser - loss_par|: {abs(loss_ser - loss_par):.3e}")

    # Compare gradients
    for k in ["load", "solar", "wind", "is_on", "is_chg", "is_dis"]:
        d = max_abs_diff(g_ser[k], g_par[k])
        if d is None:
            print(f"grad[{k}]: None (no grad)")
        else:
            print(f"Max |grad_{k} serial - parallel|: {d:.3e}")
    


if __name__ == "__main__":
    main()
