import torch
from src.models.ed_model_qp import EDModelLP
from src.models.data_classes import create_data_dict
import pandas as pd
import json
from scipy import sparse
from scipy.optimize import linprog
import numpy as np


def quick_grad_test(
    ed_model_lp, load, solar_max, wind_max, is_on, is_charging, is_discharging
):
    load = load.clone().detach().requires_grad_(True)
    solar_max = solar_max.clone().detach().requires_grad_(True)
    wind_max = wind_max.clone().detach().requires_grad_(True)
    is_on = is_on.clone().detach().requires_grad_(True)
    is_charging = is_charging.clone().detach().requires_grad_(True)
    is_discharging = is_discharging.clone().detach().requires_grad_(True)

    f = ed_model_lp.objective(
        load, solar_max, wind_max, is_on, is_charging, is_discharging
    )
    loss = f.mean()
    loss.backward()

    print("f:", f.detach().cpu().numpy())
    print("grad solar max abs:", solar_max.grad.abs().max().item())

    return load, solar_max, wind_max, is_on, is_charging, is_discharging, f


def sign_check_one(
    ed_model_lp,
    load,
    solar_max,
    wind_max,
    is_on,
    is_charging,
    is_discharging,
    t=0,
    eps=1.0,
):
    # Make sure tensors are detached (no grad) for clean FD
    load0 = load.detach().clone()
    solar0 = solar_max.detach().clone()
    wind0 = wind_max.detach().clone()
    on0 = is_on.detach().clone()
    ch0 = is_charging.detach().clone()
    dis0 = is_discharging.detach().clone()

    # Base objective
    f0 = ed_model_lp.objective(load0, solar0, wind0, on0, ch0, dis0)  # (B,) or scalar

    # Perturb solar at hour t
    solar1 = solar0.clone()
    if solar1.dim() == 1:
        solar1[t] += eps
    else:
        solar1[:, t] += eps

    f1 = ed_model_lp.objective(load0, solar1, wind0, on0, ch0, dis0)

    fd = (f1 - f0) / eps
    return f0, f1, fd


def debug_solar_row(
    ed_model_lp, load, solar_max, wind_max, is_on, is_charging, is_discharging, t=0
):
    # Build p,h,b as used by objective()
    p, h, b, load_b, solar_b, wind_b, on_b, chg_b, dis_b = ed_model_lp.build_phb(
        load, solar_max, wind_max, is_on, is_charging, is_discharging
    )

    # Find the inequality row for solar pg_ub at hour t
    solar_p = ed_model_lp.pg_idx_solar
    solar_row = None
    for row_id in ed_model_lp.builder.ub_rows.get("pg_ub", []):
        _, (p_idx, tt) = ed_model_lp.h_spec[row_id]
        if p_idx == solar_p and tt == t:
            solar_row = row_id
            break
    assert solar_row is not None, "Could not find solar pg_ub row for that t"

    # Solve ONE LP with full G (no bounds extraction) and get primal/duals
    A_ub = sparse.csr_matrix(ed_model_lp.G.detach().cpu().numpy())
    A_eq = sparse.csr_matrix(ed_model_lp.A.detach().cpu().numpy())

    c = p[0].detach().cpu().numpy().astype(np.float64, copy=False)
    h0 = h[0].detach().cpu().numpy().astype(np.float64, copy=False)
    b0 = b[0].detach().cpu().numpy().astype(np.float64, copy=False)

    res = linprog(
        c=c, A_ub=A_ub, b_ub=h0, A_eq=A_eq, b_eq=b0, bounds=None, method="highs"
    )
    if not res.success:
        raise RuntimeError(res.message)

    x = res.x
    lam = res.ineqlin.marginals  # what SciPy reports
    # Slack for row: h - (Gx)
    Gx = A_ub.dot(x)
    slack = h0[solar_row] - Gx[solar_row]

    print("solar_row_id:", solar_row)
    print("solar_max[t] used in h:", h0[solar_row])
    print("slack (h - Gx) at solar row:", slack)
    print("marginal (ineqlin) at solar row:", lam[solar_row])
    print("pred df/d solar[t] if df/dh=-lam:", -lam[solar_row])


def _get_binary_variables(output_path: str):
    with open(output_path, "r") as f:
        output_data = json.load(f)

    is_on = torch.zeros((len(output_data["Is on"]), 72))
    is_charging = torch.zeros((len(output_data["Storage discharging rates (MW)"]), 72))
    is_discharging = torch.zeros(
        (len(output_data["Storage discharging rates (MW)"]), 72)
    )

    for i, (gen, unit_status) in enumerate(
        sorted(output_data["Is on"].items(), key=lambda x: x[0])
    ):
        is_on[i, :] = torch.tensor(unit_status, dtype=torch.double)

    for i, (storage, charging_rates) in enumerate(
        sorted(output_data["Storage charging rates (MW)"].items(), key=lambda x: x[0])
    ):
        rates_tensor = torch.tensor(charging_rates, dtype=torch.double)
        is_charging[i, :] = (rates_tensor > 0).double()
    for i, (storage, discharging_rates) in enumerate(
        sorted(
            output_data["Storage discharging rates (MW)"].items(), key=lambda x: x[0]
        )
    ):
        rates_tensor = torch.tensor(discharging_rates, dtype=torch.double)
        is_discharging[i, :] = (rates_tensor > 0).double()

    assert torch.all(is_charging * is_discharging == 0), (
        "Storage units cannot charge and discharge simultaneously."
    )

    return is_on.clamp_min(0), is_charging.clamp_min(0), is_discharging.clamp_min(0)


def _get_explanatory_variables(path_features: str):
    df_profiles = pd.read_excel(path_features, sheet_name="Profiles")
    load = torch.tensor(df_profiles["demand"].values, dtype=torch.float32)
    wind_max = torch.tensor(df_profiles["wind"].values, dtype=torch.float32)
    solar_max = torch.tensor(df_profiles["solar"].values, dtype=torch.float32)

    return load, solar_max, wind_max


sample_path = "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/InputData.json"
explanatory_vars_path = "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/explanatory_variables.xlsx"
output_path = "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/OutputData.json"
data_dict = create_data_dict(sample_path)

load, solar_max, wind_max = _get_explanatory_variables(explanatory_vars_path)
is_on, is_charging, is_discharging = _get_binary_variables(output_path)

ed_model_lp = EDModelLP(data_dict)

# y = ed_model_lp(
#     load,
#     solar_max,
#     wind_max,
#     is_on,
#     is_charging,
#     is_discharging,
# )

# my_sol = unpack_lp_solution(y, ed_model_lp)

# with open("lp_solution.pkl", "wb") as f:
#     pickle.dump(my_sol, f)

load_g, solar_g, wind_g, on_g, ch_g, dis_g, f = quick_grad_test(
    ed_model_lp, load, solar_max, wind_max, is_on, is_charging, is_discharging
)

# After backward, you have solar_max.grad
t = 0
f0, f1, fd = sign_check_one(
    ed_model_lp,
    load_g,
    solar_g,
    wind_g,
    on_g,
    ch_g,
    dis_g,
    t=t,
    eps=1e-3,
)
print("float64 f0:", float(f0.double()))
print("float64 f1:", float(f1.double()))
print("float64 diff:", float((f1 - f0).double()))


print("f0:", f0.detach().cpu().numpy())
print("f1:", f1.detach().cpu().numpy())
print("FD d f / d solar[t]:", fd.detach().cpu().numpy())

print(
    "Autograd grad solar[t]:",
    (solar_g.grad[t] if solar_g.dim() == 1 else solar_g.grad[0, t])
    .detach()
    .cpu()
    .numpy(),
)

debug_solar_row(
    ed_model_lp, load, solar_max, wind_max, is_on, is_charging, is_discharging, t=t
)
