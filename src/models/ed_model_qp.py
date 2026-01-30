import torch
import torch.nn as nn
from qpth.qp import QPFunction, QPSolvers
from scipy.optimize import linprog
from scipy import sparse
import numpy as np
from torch.autograd import Function
from concurrent.futures import ProcessPoolExecutor
import os

_G_AUB = None
_G_AEQ = None
_G_OPTIONS = None


def _init_highs_worker(A_ub, A_eq, highs_time_limit):
    """
    Runs once per worker process. Stores large, constant matrices globally in that process
    so they are not pickled/sent for every LP.
    """
    global _G_AUB, _G_AEQ, _G_OPTIONS
    _G_AUB = A_ub
    _G_AEQ = A_eq
    _G_OPTIONS = (
        {"time_limit": highs_time_limit} if highs_time_limit is not None else None
    )


def _solve_highs_one(args):
    """
    Solve a single LP instance with already-initialized global A_ub, A_eq.
    args = (c, h_ub, b_eq)
    Returns tuple:
        (ok, msg, fun, lam_ub, nu_eq)
    """
    c, h_ub, b_eq = args
    res = linprog(
        c=c,
        A_ub=_G_AUB,
        b_ub=h_ub,
        A_eq=_G_AEQ,
        b_eq=b_eq,
        bounds=None,
        method="highs",
        options=_G_OPTIONS,
    )
    if not res.success:
        return (False, res.message, None, None, None)
    return (True, None, float(res.fun), res.ineqlin.marginals, res.eqlin.marginals)


class EDShapes:
    def __init__(self, G, P, S, K, T):
        self.G, self.P, self.S, self.K, self.T = int(G), int(P), int(S), int(K), int(T)


class VarIndex:
    def __init__(self, sh: EDShapes):
        self.sh = sh
        G, P, S, K, T = sh.G, sh.P, sh.S, sh.K, sh.T

        off = 0  # offset
        self.off_pg = off
        off += P * T  # profiled_generation[P,T]
        self.off_s = off
        off += S * T  # storage_level[S,T]
        self.off_cr = off
        off += S * T  # charge_rate[S,T]
        self.off_dr = off
        off += S * T  # discharge_rate[S,T]
        self.off_seg = off
        off += G * T * K  # segprod[G,T,K]
        self.off_pa = off
        off += G * T  # prod_above[G,T]
        self.off_curt = off
        off += T  # curtailment[T]
        self.nz = off  # total number of variables

    def pg(self, p, t):
        return self.off_pg + p * self.sh.T + t  # profiled_generation

    def s(self, u, t):
        return self.off_s + u * self.sh.T + t  # storage_level

    def cr(self, u, t):
        return self.off_cr + u * self.sh.T + t  # charge_rate

    def dr(self, u, t):
        return self.off_dr + u * self.sh.T + t  # discharge_rate

    def seg(self, g, t, k):
        return self.off_seg + (g * self.sh.T + t) * self.sh.K + k  # segprod

    def pa(self, g, t):
        return self.off_pa + g * self.sh.T + t  # prod_above

    def curt(self, t):
        return self.off_curt + t  # curtailment


class RowBuilder:
    """
    Dense row builder for qpth/QPFunction.
    Keeps row maps so we can fill b/h later without rebuilding A/G.
    """

    def __init__(self, nz, device="cpu", dtype=torch.float32):
        self.nz = nz
        self.device = torch.device(device)
        self.dtype = dtype

        self.A_rows = []  # each: (idxs, vals, rhs_key, meta)
        self.G_rows = []  # each: (idxs, vals, rhs_key, meta)

        self.eq_rows = {}  # rhs_key -> list[row_id]
        self.ub_rows = {}
        self.eq_meta = {}
        self.ub_meta = {}

    def add_eq_row(self, coeffs: dict, rhs_key: str, meta=None):
        row_id = len(self.A_rows)
        idxs = torch.tensor(list(coeffs.keys()), device=self.device, dtype=torch.long)
        vals = torch.tensor(list(coeffs.values()), device=self.device, dtype=self.dtype)
        self.A_rows.append((idxs, vals, rhs_key, meta))
        self.eq_rows.setdefault(rhs_key, []).append(row_id)
        self.eq_meta.setdefault(rhs_key, []).append(meta)
        return row_id

    def add_ub_row(self, coeffs: dict, rhs_key: str, meta=None):
        row_id = len(self.G_rows)
        idxs = torch.tensor(list(coeffs.keys()), device=self.device, dtype=torch.long)
        vals = torch.tensor(list(coeffs.values()), device=self.device, dtype=self.dtype)
        self.G_rows.append((idxs, vals, rhs_key, meta))
        self.ub_rows.setdefault(rhs_key, []).append(row_id)
        self.ub_meta.setdefault(rhs_key, []).append(meta)
        return row_id

    def finalize(self):
        neq = len(self.A_rows)
        nineq = len(self.G_rows)

        A = torch.zeros((neq, self.nz), device=self.device, dtype=self.dtype)
        G = torch.zeros((nineq, self.nz), device=self.device, dtype=self.dtype)

        for r, (idxs, vals, _, _) in enumerate(self.A_rows):
            A[r, idxs] = vals
        for r, (idxs, vals, _, _) in enumerate(self.G_rows):
            G[r, idxs] = vals

        # specs in row order
        b_spec = [(rhs_key, meta) for (_, _, rhs_key, meta) in self.A_rows]
        h_spec = [(rhs_key, meta) for (_, _, rhs_key, meta) in self.G_rows]
        return A, G, b_spec, h_spec


class EDModelQP(nn.Module):
    def __init__(self, ed_data_dict, eps=1e-3, device="cpu"):
        super().__init__()
        self.eps = eps
        self.device = torch.device(device)
        T = len(ed_data_dict["system_data"].load)
        G = len(ed_data_dict["thermal_gen_data_list"])
        P = len(ed_data_dict["profiled_gen_data_list"])
        S = len(ed_data_dict["storage_data_list"])
        K = max(
            len(g.production_cost_curve) for g in ed_data_dict["thermal_gen_data_list"]
        )

        self.sh = EDShapes(G=G, P=P, S=S, K=K, T=T)
        self.idx = VarIndex(self.sh)
        nz = self.idx.nz

        # Initialize p
        p0 = torch.zeros(nz, device=self.device)
        self.register_buffer("p0", p0)

        # Set up Q, eps * I
        Q = self.eps * torch.eye(nz, device=self.device)
        self.register_buffer("Q", Q)

        # Build constant matrices A, G
        builder = RowBuilder(nz=nz, device=self.device)
        self.builder = builder

        self._add_profiled_gen_rows(builder, ed_data_dict)
        self._add_storage_rows(builder, ed_data_dict)
        self._add_thermal_rows(builder, ed_data_dict)
        self._add_system_rows(builder, ed_data_dict)

        A, G, b_spec, h_spec = builder.finalize()
        self.neq = A.shape[0]
        self.nineq = G.shape[0]
        self.register_buffer("A", A)
        self.register_buffer("G", G)

        # Store rhs specs for later filling during forward
        self.b_spec = b_spec
        self.h_spec = h_spec

    def forward(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        verbose=False,
    ):
        p, h, b, *_ = self.build_phb(
            load, solar_max, wind_max, is_on, is_charging, is_discharging
        )
        y = QPFunction(verbose=verbose, solver=QPSolvers.PDIPM_BATCHED)(
            self.Q, p, self.G, h, self.A, b
        )
        return y  # you should probably have this be a dict

    # -- helper functions --

    def _add_profiled_gen_rows(self, builder: RowBuilder, ed_data_dict):
        sh = self.sh
        idx = self.idx

        profiled = sorted(ed_data_dict["profiled_gen_data_list"], key=lambda g: g.name)
        names = [g.name for g in profiled]

        self.profiled_units_names = names
        min_power = torch.tensor(
            [g.min_power for g in profiled], device=builder.device, dtype=builder.dtype
        )
        max_power = torch.tensor(
            [g.max_power for g in profiled], device=builder.device, dtype=builder.dtype
        )

        self.pg_idx_solar = names.index("solar")
        self.pg_idx_wind = names.index("wind")

        cost = torch.tensor(
            [g.cost for g in profiled], device=builder.device, dtype=builder.dtype
        )

        for p in range(sh.P):  # TODO: think about vectorizing this later
            for t in range(sh.T):
                # Add gen cost
                self.p0[idx.pg(p, t)] = cost[p]

                # Adding pmin constraints: - pg[p, t] <= - min_power[p]
                builder.add_ub_row({idx.pg(p, t): -1.0}, rhs_key="pg_lb", meta=(p, t))

                # Adding pmax constraints: pg[p, t] <= max_power[p]
                builder.add_ub_row({idx.pg(p, t): 1.0}, rhs_key="pg_ub", meta=(p, t))

        self.register_buffer("pg_min_power", min_power)  # (P,)
        self.register_buffer("pg_max_power", max_power)  # (P,)

    def _add_storage_rows(self, builder: RowBuilder, ed_data_dict):
        sh = self.sh
        idx = self.idx
        storage = sorted(ed_data_dict["storage_data_list"], key=lambda s: s.name)
        names = [s.name for s in storage]

        self.storage_units_names = names

        max_levels = torch.tensor(
            [s.max_level for s in storage], device=builder.device, dtype=builder.dtype
        )
        min_levels = torch.tensor(
            [getattr(d, "min_level", 0.0) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

        min_charge = torch.tensor(
            [getattr(d, "min_charge_rate", 0.0) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)
        max_charge = torch.tensor(
            [d.max_charge_rate for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)

        min_discharge = torch.tensor(
            [getattr(d, "min_discharge_rate", 0.0) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)
        max_discharge = torch.tensor(
            [d.max_discharge_rate for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)

        init_levels = torch.tensor(
            [d.initial_level for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)
        loss_factors = torch.tensor(
            [getattr(d, "loss_factor", 0.0) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)

        charge_eff = torch.tensor(
            [d.charge_efficiency for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)
        discharge_eff = torch.tensor(
            [d.discharge_efficiency for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)

        min_end = torch.tensor(
            [
                getattr(d, "min_ending_level", getattr(d, "min_level", 0.0))
                for d in storage
            ],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)
        max_end = torch.tensor(
            [getattr(d, "max_ending_level", d.max_level) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )  # (U,)

        charge_costs = torch.tensor(
            [data.charge_cost for data in storage],
            device=builder.device,
            dtype=builder.dtype,
        )
        discharge_costs = torch.tensor(
            [data.discharge_cost for data in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

        self.register_buffer("st_max_levels", max_levels)
        self.register_buffer("st_min_levels", min_levels)
        self.register_buffer("st_min_charge", min_charge)
        self.register_buffer("st_max_charge", max_charge)
        self.register_buffer("st_min_discharge", min_discharge)
        self.register_buffer("st_max_discharge", max_discharge)
        self.register_buffer("st_init_levels", init_levels)
        self.register_buffer("st_loss_factors", loss_factors)
        self.register_buffer("st_charge_eff", charge_eff)
        self.register_buffer("st_discharge_eff", discharge_eff)
        self.register_buffer("st_min_end", min_end)
        self.register_buffer("st_max_end", max_end)

        # Inequality constraints
        for s in range(sh.S):
            for t in range(sh.T):
                # Add charge and discharge costs
                self.p0[idx.cr(s, t)] = charge_costs[s]
                self.p0[idx.dr(s, t)] = discharge_costs[s]

                # Storage level constraints
                builder.add_ub_row(
                    {idx.s(s, t): 1.0}, rhs_key="st_max_level", meta=(s, t)
                )
                builder.add_ub_row(
                    {idx.s(s, t): -1.0}, rhs_key="st_min_level", meta=(s, t)
                )

                # Charge rate constraints
                builder.add_ub_row(
                    {idx.cr(s, t): 1.0}, rhs_key="st_max_charge", meta=(s, t)
                )
                builder.add_ub_row(
                    {idx.cr(s, t): -1.0}, rhs_key="st_min_charge", meta=(s, t)
                )

                # Discharge rate constraints
                builder.add_ub_row(
                    {idx.dr(s, t): 1.0}, rhs_key="st_max_discharge", meta=(s, t)
                )
                builder.add_ub_row(
                    {idx.dr(s, t): -1.0}, rhs_key="st_min_discharge", meta=(s, t)
                )

            # End of horizon bounds
            builder.add_ub_row({idx.s(s, sh.T - 1): 1.0}, rhs_key="st_max_end", meta=s)
            builder.add_ub_row({idx.s(s, sh.T - 1): -1.0}, rhs_key="st_min_end", meta=s)

        # Equality constraints: storage level
        for s in range(sh.S):
            for t in range(sh.T):
                coeffs = {}
                if t == 0:
                    coeffs[idx.s(s, t)] = 1.0
                    coeffs[idx.cr(s, t)] = -charge_eff[s]
                    coeffs[idx.dr(s, t)] = 1.0 / discharge_eff[s]
                    rhs_key = "st_level_init"
                else:
                    coeffs[idx.s(s, t)] = 1.0
                    coeffs[idx.s(s, t - 1)] = -(1.0 - loss_factors[s])
                    coeffs[idx.cr(s, t)] = -charge_eff[s]
                    coeffs[idx.dr(s, t)] = 1.0 / discharge_eff[s]
                    rhs_key = "st_level_evol"

                builder.add_eq_row(coeffs, rhs_key=rhs_key, meta=(s, t))

    def _add_thermal_rows(self, builder, ed_data_dict):  # TODO: check the segment logic
        sh = self.sh
        idx = self.idx
        thermals = sorted(ed_data_dict["thermal_gen_data_list"], key=lambda g: g.name)
        G, T, K = sh.G, sh.T, sh.K

        self.thermal_units_names = [g.name for g in thermals]

        # ---- constants ----
        min_power = torch.tensor(
            [g.min_power for g in thermals], device=builder.device, dtype=builder.dtype
        )  # (G,)
        max_power = torch.tensor(
            [g.max_power for g in thermals], device=builder.device, dtype=builder.dtype
        )  # (G,)

        power_diff = torch.clamp(max_power - min_power, min=0.0)  # (G,)
        power_diff = torch.where(
            power_diff < 1e-7, torch.zeros_like(power_diff), power_diff
        )

        # Build segment_mw and segment_cost as (G,K) with zero-padding
        seg_mw = torch.zeros((G, K), device=self.device)
        seg_cost = torch.zeros((G, K), device=self.device)
        for gi, g in enumerate(thermals):
            curve = g.production_cost_curve  # list of (mw, cost)
            n = len(curve)
            seg_mw[gi, :n] = torch.tensor([mw for mw, _ in curve], device=self.device)
            seg_cost[gi, :n] = torch.tensor([c for _, c in curve], device=self.device)

        # Save as buffers for RHS fill + objective
        self.register_buffer("th_min_power", min_power)
        self.register_buffer("th_max_power", max_power)
        self.register_buffer("th_power_diff", power_diff)
        self.register_buffer("th_seg_mw", seg_mw)  # (G,K)
        self.register_buffer("th_seg_cost", seg_cost)  # (G,K)

        # segment_costs = sum_{g,t,k} seg_cost[g,k] * segprod[g,t,k]
        for g in range(G):
            for t in range(T):
                for k in range(K):
                    self.p0[idx.seg(g, t, k)] += seg_cost[g, k]

        # -------------------------
        # Inequalities: G z <= h
        # -------------------------

        # (1) prod_above[g,t] <= power_diff[g] * is_on[g,t]
        for g in range(G):
            for t in range(T):
                builder.add_ub_row(
                    {idx.pa(g, t): 1.0},
                    rhs_key="pa_ub_on",
                    meta=(g, t),
                )

        # (2) segprod[g,t,k] <= seg_mw[g,k] * is_on[g,t]
        for g in range(G):
            for t in range(T):
                for k in range(K):
                    builder.add_ub_row(
                        {idx.seg(g, t, k): 1.0},
                        rhs_key="seg_ub_on",
                        meta=(g, t, k),
                    )

        # Nonnegativity explicitly:
        # segprod >= 0  -> -segprod <= 0
        # prod_above >= 0 -> -prod_above <= 0
        for g in range(G):
            for t in range(T):
                builder.add_ub_row({idx.pa(g, t): -1.0}, rhs_key="pa_nn", meta=(g, t))
                for k in range(K):
                    builder.add_ub_row(
                        {idx.seg(g, t, k): -1.0}, rhs_key="seg_nn", meta=(g, t, k)
                    )

        # -------------------------
        # Equalities: A z = b
        # -------------------------

        # (4) prod_above[g,t] == sum_k segprod[g,t,k]
        # -> prod_above[g,t] - sum_k segprod[g,t,k] = 0
        for g in range(G):
            for t in range(T):
                coeffs = {idx.pa(g, t): 1.0}
                for k in range(K):
                    coeffs[idx.seg(g, t, k)] = -1.0
                builder.add_eq_row(coeffs, rhs_key="pa_link", meta=(g, t))

    def _add_system_rows(self, builder, ed_data_dict):
        sh = self.sh
        idx = self.idx

        power_balance_penalty = ed_data_dict["system_data"].power_balance_penalty
        self.power_balance_penalty = power_balance_penalty

        # Equality: power balance
        for t in range(sh.T):
            # Add curtailment cost
            self.p0[idx.curt(t)] = power_balance_penalty

            coeffs = {}

            # Profiled generation
            for p in range(sh.P):
                coeffs[idx.pg(p, t)] = 1.0

            # Thermal generation
            for g in range(sh.G):
                coeffs[idx.pa(g, t)] = 1.0

            # Storage discharge
            for s in range(sh.S):
                coeffs[idx.dr(s, t)] = 1.0

            # Storage charge
            for s in range(sh.S):
                coeffs[idx.cr(s, t)] = -1.0

            # Curtailment
            coeffs[idx.curt(t)] = 1.0

            builder.add_eq_row(coeffs, rhs_key="power_balance", meta=t)

        # Upper bound on curtailment: curt(t) <= load(t)
        # Lower bound on curtailment is zero by variable nonnegativity
        for t in range(sh.T):
            builder.add_ub_row({idx.curt(t): 1.0}, rhs_key="curt_ub", meta=t)
            builder.add_ub_row({idx.curt(t): -1.0}, rhs_key="curt_nn", meta=t)

    def _fill_profiled_rhs(self, h, solar_max, wind_max):
        # pg lower bounds
        for row_id in self.builder.ub_rows.get("pg_lb", []):
            rhs_key, meta = self.h_spec[row_id]
            p, t = meta
            h[:, row_id] = -self.pg_min_power[p][t]

        # pg upper bounds
        for row_id in self.builder.ub_rows.get("pg_ub", []):
            rhs_key, meta = self.h_spec[row_id]
            p, t = meta
            if p == self.pg_idx_solar:
                h[:, row_id] = solar_max[:, t]
            elif p == self.pg_idx_wind:
                h[:, row_id] = wind_max[:, t]
            else:
                h[:, row_id] = self.pg_max_power[p][t]

    def _fill_storage_rhs(self, b, h, is_charging, is_discharging):
        # -------- Inequalities --------

        for row_id in self.builder.ub_rows.get("st_max_level", []):
            _, (s, t) = self.h_spec[row_id]
            h[:, row_id] = self.st_max_levels[s]

        for row_id in self.builder.ub_rows.get("st_min_level", []):
            _, (s, t) = self.h_spec[row_id]
            h[:, row_id] = -self.st_min_levels[s]

        for row_id in self.builder.ub_rows.get("st_max_charge", []):
            _, (s, t) = self.h_spec[row_id]
            h[:, row_id] = self.st_max_charge[s] * is_charging[:, s, t]

        for row_id in self.builder.ub_rows.get("st_min_charge", []):
            _, (s, t) = self.h_spec[row_id]
            h[:, row_id] = -self.st_min_charge[s] * is_charging[:, s, t]

        for row_id in self.builder.ub_rows.get("st_max_discharge", []):
            _, (s, t) = self.h_spec[row_id]
            h[:, row_id] = self.st_max_discharge[s] * is_discharging[:, s, t]

        for row_id in self.builder.ub_rows.get("st_min_discharge", []):
            _, (s, t) = self.h_spec[row_id]
            h[:, row_id] = -self.st_min_discharge[s] * is_discharging[:, s, t]

        for row_id in self.builder.ub_rows.get("st_max_end", []):
            _, s = self.h_spec[row_id]
            h[:, row_id] = self.st_max_end[s]

        for row_id in self.builder.ub_rows.get("st_min_end", []):
            _, s = self.h_spec[row_id]
            h[:, row_id] = -self.st_min_end[s]

        # -------- Equalities --------

        for row_id in self.builder.eq_rows.get("st_level_init", []):
            _, (s, t) = self.b_spec[row_id]
            b[:, row_id] = self.st_init_levels[s]

        for row_id in self.builder.eq_rows.get("st_level_evol", []):
            b[:, row_id] = 0.0

    def _fill_thermal_rhs(self, b, h, is_on):
        # ---- Inequalities ----

        # pa(g,t) <= power_diff[g] * is_on(g,t)
        for row_id in self.builder.ub_rows.get("pa_ub_on", []):
            _, (g, t) = self.h_spec[row_id]
            h[:, row_id] = self.th_power_diff[g] * is_on[:, g, t]

        # seg(g,t,k) <= seg_mw[g,k] * is_on(g,t)
        for row_id in self.builder.ub_rows.get("seg_ub_on", []):
            _, (g, t, k) = self.h_spec[row_id]
            h[:, row_id] = self.th_seg_mw[g, k] * is_on[:, g, t]

        # Nonnegativity: -x <= 0
        for row_id in self.builder.ub_rows.get("pa_nn", []):
            h[:, row_id] = 0.0

        for row_id in self.builder.ub_rows.get("seg_nn", []):
            h[:, row_id] = 0.0

        # ---- Equalities ----

        # pa(g,t) - sum_k seg(g,t,k) = 0
        for row_id in self.builder.eq_rows.get("pa_link", []):
            b[:, row_id] = 0.0

    def _fill_system_rhs(self, b, h, load, is_on):
        # Power balance
        for row_id in self.builder.eq_rows.get("power_balance", []):
            _, t = self.b_spec[row_id]

            # subtract fixed thermal minimum output from RHS
            # shape: (B,)
            pmin_on = (self.th_min_power[None, :] * is_on[:, :, t]).sum(dim=1)

            b[:, row_id] = load[:, t] - pmin_on

        # Curtailment ub
        for row_id in self.builder.ub_rows.get("curt_ub", []):
            _, t = self.h_spec[row_id]
            h[:, row_id] = load[:, t]

        for row_id in self.builder.ub_rows.get("curt_nn", []):
            h[:, row_id] = 0.0

    def _make_batched(
        self, load, solar_max, wind_max, is_on, is_charging, is_discharging
    ):
        if load.dim() == 1:
            B = 1
        else:
            B = load.shape[0]  # batch size

        if solar_max.dim() == 1:
            solar_max = solar_max.unsqueeze(0).expand(B, -1)
        if wind_max.dim() == 1:
            wind_max = wind_max.unsqueeze(0).expand(B, -1)
        if load.dim() == 1:
            load = load.unsqueeze(0).expand(B, -1)
        if is_on.dim() == 2:
            is_on = is_on.unsqueeze(0).expand(B, -1, -1)
        if is_charging.dim() == 2:
            is_charging = is_charging.unsqueeze(0).expand(B, -1, -1)
        if is_discharging.dim() == 2:
            is_discharging = is_discharging.unsqueeze(0).expand(B, -1, -1)

        return load, solar_max, wind_max, is_on, is_charging, is_discharging

    def build_phb(self, load, solar_max, wind_max, is_on, is_charging, is_discharging):
        """
        Build the (p, h, b) RHS for the QP/LP based on the current inputs.
        Returns:
            p: (B, nz)
            h: (B, nineq)
            b: (B, neq)
        """
        load, solar_max, wind_max, is_on, is_charging, is_discharging = (
            self._make_batched(
                load, solar_max, wind_max, is_on, is_charging, is_discharging
            )
        )

        B = load.shape[0]
        h = torch.zeros((B, self.nineq), device=self.device)
        b = torch.zeros((B, self.neq), device=self.device)
        p = self.p0.unsqueeze(0).expand(B, -1)  # (B, nz)

        self._fill_profiled_rhs(h, solar_max, wind_max)
        self._fill_storage_rhs(b, h, is_charging, is_discharging)
        self._fill_thermal_rhs(b, h, is_on)
        self._fill_system_rhs(b, h, load, is_on)

        return p, h, b, load, solar_max, wind_max, is_on, is_charging, is_discharging


class EDModelLP(EDModelQP):
    """
    LP sanity-check solver for the ED formulation using SciPy HiGHS.

    Reuses all matrix construction and RHS filling from EDModelQP, but solves:
        min_y p^T y
        s.t.  G y <= h
              A y  = b

    To make this scalable, it extracts all single-variable +/-1 inequality rows in G
    into variable bounds lb <= y <= ub, and removes those rows from (G,h) before calling HiGHS.
    """

    def __init__(
        self,
        ed_data_dict,
        device="cpu",
        extract_bounds=True,
        use_sparse=True,
        parallel_solve=True,
        lp_workers=None,
        parallel_min_batch=8,
    ):
        super().__init__(ed_data_dict, eps=0.0, device=device)  # eps not used for LP
        self.extract_bounds = extract_bounds
        self.use_sparse = use_sparse

        self.parallel_solve = bool(parallel_solve)
        self.lp_workers = lp_workers
        self.parallel_min_batch = int(parallel_min_batch)

        # Precompute which inequality rows in G are pure bounds (single nonzero, coeff = +1 or -1)
        with torch.no_grad():
            G = self.G  # (nineq, nz)
            nz = G.shape[1]

            nz_mask = G != 0
            nnz = nz_mask.sum(dim=1)  # (nineq,)
            single = nnz == 1

            row_ids = torch.nonzero(single, as_tuple=False).flatten()
            if row_ids.numel() == 0:
                self.bound_row_ids = None
                self.bound_cols = None
                self.bound_sign = None
                self.keep_row_mask = torch.ones(
                    (self.nineq,), dtype=torch.bool, device=G.device
                )
            else:
                rc = torch.nonzero(
                    nz_mask[row_ids], as_tuple=False
                )  # (k, 2): [row_in_subset, col]
                cols = rc[:, 1]  # (k,)
                coeff = G[row_ids, cols]  # (k,)

                is_pos1 = torch.isclose(coeff, torch.ones_like(coeff))
                is_neg1 = torch.isclose(coeff, -torch.ones_like(coeff))
                is_pm1 = is_pos1 | is_neg1

                # Only treat exact +/-1 singleton rows as bounds; keep the others in G.
                bound_row_ids = row_ids[is_pm1]
                bound_cols = cols[is_pm1]
                bound_sign = torch.where(
                    is_pos1[is_pm1],
                    torch.ones_like(bound_cols),
                    -torch.ones_like(bound_cols),
                ).to(G.dtype)

                keep_mask = torch.ones((self.nineq,), dtype=torch.bool, device=G.device)
                keep_mask[bound_row_ids] = False

                self.bound_row_ids = bound_row_ids
                self.bound_cols = bound_cols
                self.bound_sign = bound_sign
                self.keep_row_mask = keep_mask

            # Cache reduced G (rows kept) in CPU SciPy-friendly format
            G_keep = self.G[self.keep_row_mask, :]  # (nkeep, nz)
            G_cpu = G_keep.detach().cpu().numpy()

            if self.use_sparse:
                self._G_keep_scipy = sparse.csr_matrix(G_cpu)
            else:
                self._G_keep_scipy = G_cpu

            # Cache A once too
            A_cpu = self.A.detach().cpu().numpy()
            if self.use_sparse:
                self._A_scipy = sparse.csr_matrix(A_cpu)
            else:
                self._A_scipy = A_cpu

            self.nz_lp = nz  # convenience

            # ---- EXISTING: cache reduced G for the non-diff forward() path ----
            G_keep = self.G[self.keep_row_mask, :]  # (nkeep, nz)
            G_cpu = G_keep.detach().cpu().numpy()
            self._G_keep_scipy = sparse.csr_matrix(G_cpu) if self.use_sparse else G_cpu

            # ---- EXISTING: cache A ----
            A_cpu = self.A.detach().cpu().numpy()
            self._A_scipy = sparse.csr_matrix(A_cpu) if self.use_sparse else A_cpu

            # ---- NEW: cache FULL G too (for differentiable objective) ----
            G_full_cpu = self.G.detach().cpu().numpy()
            self._G_full_scipy = (
                sparse.csr_matrix(G_full_cpu) if self.use_sparse else G_full_cpu
            )

    def forward(  # why do you even have this here?
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        *,
        return_scipy_results=False,
        highs_time_limit=None,
    ):
        p, h, b, load, *_ = self.build_phb(
            load, solar_max, wind_max, is_on, is_charging, is_discharging
        )

        B = load.shape[0]

        # Solve one LP per batch item (HiGHS is not batched)
        ys = []
        results = []

        # Precompute CPU objective (same each batch item)
        # If you want per-instance objective later, replace with a computed p.
        # (Right now p is constant, which matches your code.)
        for i in range(B):
            c = p[i].detach().cpu().numpy().astype(np.float64, copy=False)
            b_eq = b[i].detach().cpu().numpy().astype(np.float64, copy=False)

            if self.extract_bounds and (self.bound_row_ids is not None):
                # Start with (-inf, +inf) bounds
                lb = np.full((self.nz_lp,), -np.inf, dtype=np.float64)
                ub = np.full((self.nz_lp,), np.inf, dtype=np.float64)

                # Pull out the bound RHS values for this instance
                h_i = h[i]  # (nineq,) on self.device
                rhs = (
                    h_i[self.bound_row_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                cols = self.bound_cols.detach().cpu().numpy()
                signs = (
                    self.bound_sign.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )

                # +1 * y_j <= u  => ub[j] = min(ub[j], u)
                pos = signs > 0
                if np.any(pos):
                    j = cols[pos]
                    u = rhs[pos]
                    # multiple constraints can hit same j
                    for jj, uu in zip(j, u):
                        if uu < ub[jj]:
                            ub[jj] = uu

                # -1 * y_j <= -l => y_j >= l  => lb[j] = max(lb[j], l) where l = -rhs
                neg = signs < 0
                if np.any(neg):
                    j = cols[neg]
                    l = -rhs[neg]
                    for jj, ll in zip(j, l):
                        if ll > lb[jj]:
                            lb[jj] = ll

                # Remaining inequalities
                h_ub = (
                    h_i[self.keep_row_mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                A_ub = self._G_keep_scipy

                bounds = list(zip(lb.tolist(), ub.tolist()))
            else:
                # ---- CHANGED: use cached full G (avoid rebuilding csr each call) ----
                A_ub = self._G_full_scipy
                h_ub = h[i].detach().cpu().numpy().astype(np.float64, copy=False)
                bounds = None

            res = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=h_ub,
                A_eq=self._A_scipy,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options=(
                    {"time_limit": highs_time_limit}
                    if highs_time_limit is not None
                    else None
                ),
            )

            results.append(res)
            if not res.success:
                raise RuntimeError(f"HiGHS LP failed (batch {i}): {res.message}")

            ys.append(torch.tensor(res.x, device=self.device, dtype=torch.float32))

        y = torch.stack(ys, dim=0)  # (B, nz)

        return (y, results) if return_scipy_results else y

    def objective(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        highs_time_limit=None,
    ):
        return EDLPObjectiveFn.apply(
            self,
            load,
            solar_max,
            wind_max,
            is_on,
            is_charging,
            is_discharging,
            highs_time_limit,
        )


class EDLPObjectiveFn(Function):
    @staticmethod
    def forward(
        ctx,
        model: EDModelLP,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        highs_time_limit=None,
    ):
        """
        Returns f*: (B,) optimal objective value(s).
        Parallelizes the B independent HiGHS solves across processes when beneficial.
        """
        p, h, b, load, solar_max, wind_max, is_on, is_charging, is_discharging = (
            model.build_phb(
                load, solar_max, wind_max, is_on, is_charging, is_discharging
            )
        )
        B = load.shape[0]

        # ---- CHANGED: reuse cached sparse matrices (avoid rebuilding csr every forward) ----
        A_ub = model._G_full_scipy
        A_eq = model._A_scipy

        # ---- CHANGED: convert to numpy ONCE ----
        c_np = p.detach().cpu().numpy().astype(np.float64, copy=False)  # (B,nz)
        h_np = h.detach().cpu().numpy().astype(np.float64, copy=False)  # (B,nineq)
        b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)  # (B,neq)

        # Prepare tasks for each batch item
        tasks = [(c_np[i], h_np[i], b_np[i]) for i in range(B)]

        f_list = []
        lam_ub_list = []
        nu_eq_list = []

        do_parallel = (
            model.parallel_solve
            and B >= model.parallel_min_batch
            and (os.cpu_count() or 1) > 1
        )

        if not do_parallel:
            # ---- SERIAL fallback ----
            for i in range(B):
                res = linprog(
                    c=tasks[i][0],
                    A_ub=A_ub,
                    b_ub=tasks[i][1],
                    A_eq=A_eq,
                    b_eq=tasks[i][2],
                    bounds=None,
                    method="highs",
                    options=(
                        {"time_limit": highs_time_limit}
                        if highs_time_limit is not None
                        else None
                    ),
                )
                if not res.success:
                    raise RuntimeError(f"HiGHS LP failed (batch {i}): {res.message}")
                f_list.append(res.fun)
                lam_ub_list.append(res.ineqlin.marginals)
                nu_eq_list.append(res.eqlin.marginals)
        else:
            # ---- PARALLEL solve across batch items ----
            max_workers = model.lp_workers
            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, B)

            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_highs_worker,
                initargs=(A_ub, A_eq, highs_time_limit),
            ) as ex:
                outs = list(ex.map(_solve_highs_one, tasks))

            for i, (ok, msg, fun, lam, nu) in enumerate(outs):
                if not ok:
                    raise RuntimeError(f"HiGHS LP failed (batch {i}): {msg}")
                f_list.append(fun)
                lam_ub_list.append(lam)
                nu_eq_list.append(nu)

        device = model.device
        dtype = load.dtype

        f = torch.as_tensor(np.array(f_list), device=device, dtype=dtype)  # (B,)
        lam_ub = torch.as_tensor(
            np.stack(lam_ub_list), device=device, dtype=dtype
        )  # (B,nineq)
        nu_eq = torch.as_tensor(
            np.stack(nu_eq_list), device=device, dtype=dtype
        )  # (B,neq)

        # Save for backward
        ctx.model = model
        ctx.save_for_backward(
            lam_ub, nu_eq, load, solar_max, wind_max, is_on, is_charging, is_discharging
        )

        # shape flags (same as your original)
        ctx.load_was_1d = load.dim() == 1
        ctx.solar_was_1d = solar_max.dim() == 1
        ctx.wind_was_1d = wind_max.dim() == 1
        ctx.is_on_was_2d = is_on.dim() == 2
        ctx.is_chg_was_2d = is_charging.dim() == 2
        ctx.is_dis_was_2d = is_discharging.dim() == 2

        return f

    @staticmethod
    def backward(ctx, grad_out):
        # ---- UNCHANGED from your version (kept as-is) ----
        model = ctx.model
        lam_ub, nu_eq, load, solar_max, wind_max, is_on, is_charging, is_discharging = (
            ctx.saved_tensors
        )

        if grad_out.dim() == 0:
            grad_out = grad_out.unsqueeze(0)
        go = grad_out.view(-1, 1)

        df_dh = lam_ub * go  # (B,nineq)
        df_db = nu_eq * go  # (B,neq)

        g_load = torch.zeros_like(load)
        g_solar = torch.zeros_like(solar_max)
        g_wind = torch.zeros_like(wind_max)
        g_is_on = torch.zeros_like(is_on)
        g_is_chg = torch.zeros_like(is_charging)
        g_is_dis = torch.zeros_like(is_discharging)

        for row_id in model.builder.ub_rows.get("pg_ub", []):
            _, (p, t) = model.h_spec[row_id]
            if p == model.pg_idx_solar:
                g_solar[:, t] += df_dh[:, row_id]
            elif p == model.pg_idx_wind:
                g_wind[:, t] += df_dh[:, row_id]

        for row_id in model.builder.ub_rows.get("st_max_charge", []):
            _, (s, t) = model.h_spec[row_id]
            g_is_chg[:, s, t] += df_dh[:, row_id] * model.st_max_charge[s]

        for row_id in model.builder.ub_rows.get("st_min_charge", []):
            _, (s, t) = model.h_spec[row_id]
            g_is_chg[:, s, t] += df_dh[:, row_id] * (-model.st_min_charge[s])

        for row_id in model.builder.ub_rows.get("st_max_discharge", []):
            _, (s, t) = model.h_spec[row_id]
            g_is_dis[:, s, t] += df_dh[:, row_id] * model.st_max_discharge[s]

        for row_id in model.builder.ub_rows.get("st_min_discharge", []):
            _, (s, t) = model.h_spec[row_id]
            g_is_dis[:, s, t] += df_dh[:, row_id] * (-model.st_min_discharge[s])

        for row_id in model.builder.ub_rows.get("pa_ub_on", []):
            _, (g, t) = model.h_spec[row_id]
            g_is_on[:, g, t] += df_dh[:, row_id] * model.th_power_diff[g]

        for row_id in model.builder.ub_rows.get("seg_ub_on", []):
            _, (g, t, k) = model.h_spec[row_id]
            g_is_on[:, g, t] += df_dh[:, row_id] * model.th_seg_mw[g, k]

        for row_id in model.builder.ub_rows.get("curt_ub", []):
            _, t = model.h_spec[row_id]
            g_load[:, t] += df_dh[:, row_id]

        for row_id in model.builder.eq_rows.get("power_balance", []):
            _, t = model.b_spec[row_id]
            g_load[:, t] += df_db[:, row_id]
            g_is_on[:, :, t] += df_db[:, row_id].unsqueeze(1) * (
                -model.th_min_power[None, :]
            )

        if ctx.load_was_1d:
            g_load = g_load.squeeze(0)
        if ctx.solar_was_1d:
            g_solar = g_solar.squeeze(0)
        if ctx.wind_was_1d:
            g_wind = g_wind.squeeze(0)
        if ctx.is_on_was_2d:
            g_is_on = g_is_on.squeeze(0)
        if ctx.is_chg_was_2d:
            g_is_chg = g_is_chg.squeeze(0)
        if ctx.is_dis_was_2d:
            g_is_dis = g_is_dis.squeeze(0)

        return (None, g_load, g_solar, g_wind, g_is_on, g_is_chg, g_is_dis, None)
