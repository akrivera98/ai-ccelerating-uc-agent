from src.registry import registry
import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from scipy.optimize import linprog
from scipy import sparse
from joblib import Parallel, delayed

from src.ed_models.cannon import EDFormulation, EDRHSBuilder
from src.ed_models.data_utils import create_data_dict
# ============================================================
# Multiprocessing helpers for HiGHS
# ============================================================


def _solve_highs_one(c, h_ub, b_eq, bounds, A_ub, A_eq, options):
    """
    Returns (ok, msg, fun, x, lam_ub, nu_eq)
    """
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=h_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not res.success:
        return (False, res.message, None, None, None, None)

    return (
        True,
        None,
        float(res.fun),
        res.x,
        res.ineqlin.marginals,
        res.eqlin.marginals,
    )


def _solve_highs_chunk(idxs, c_np, h_np, b_np, A_ub, A_eq, options):
    """
    Solve a chunk of LPs identified by `idxs` (list/array of batch indices).

    Returns:
      idxs_out, ok, msg, funs, lams, nus
    where funs/lams/nus are aligned with idxs_out.
    """
    funs = []
    lams = []
    nus = []
    for i in idxs:
        res = linprog(
            c=c_np[i],
            A_ub=A_ub,
            b_ub=h_np[i],
            A_eq=A_eq,
            b_eq=b_np[i],
            bounds=None,  # full-G path; keep consistent
            method="highs",
            options=options,
        )
        if not res.success:
            return (idxs, False, f"batch {i}: {res.message}", None, None, None)

        funs.append(float(res.fun))
        lams.append(res.ineqlin.marginals)
        nus.append(res.eqlin.marginals)

    return (idxs, True, None, funs, lams, nus)


# ============================================================
# SciPy / HiGHS solver adapter (non-differentiable primal solve)
# ============================================================


class EDHiGHSSolver(nn.Module):
    """
    Component 3b: SciPy HiGHS solver adapter (for primal x).

    - Reuses compiled matrices from EDFormulation (form.A, form.G)
    - Reuses EDRHSBuilder to build batched q,h,b
    - Optionally extracts singleton +/-1 rows in G into variable bounds for speed

    IMPORTANT:
      - This class is for getting primal solutions x.
      - For differentiable objective + manual backward, use EDHiGHSObjectiveFn below.
    """

    def __init__(
        self,
        form,
        rhs_builder: nn.Module,
        *,
        extract_bounds: bool = True,
        use_sparse: bool = True,
    ):
        super().__init__()
        self.form = form
        self.rhs = rhs_builder
        self.extract_bounds = bool(extract_bounds)
        self.use_sparse = bool(use_sparse)

        # Cache SciPy-friendly A (constant)
        A_cpu = form.A.detach().cpu().numpy()
        self._A_scipy = sparse.csr_matrix(A_cpu) if self.use_sparse else A_cpu

        # Cache G (both full and reduced-if-bounds)
        G_full_cpu = form.G.detach().cpu().numpy()
        self._G_full_scipy = (
            sparse.csr_matrix(G_full_cpu) if self.use_sparse else G_full_cpu
        )

        # Precompute bound extraction masks from FULL G (on CPU / torch)
        if self.extract_bounds:
            self._precompute_bounds()

    def _precompute_bounds(self):
        with torch.no_grad():
            G = self.form.G  # (nineq, nz)
            nz_mask = G != 0
            nnz = nz_mask.sum(dim=1)
            single = nnz == 1
            row_ids = torch.nonzero(single, as_tuple=False).flatten()

            if row_ids.numel() == 0:
                self.bound_row_ids = None
                self.bound_cols = None
                self.bound_sign = None
                self.keep_row_mask = torch.ones(
                    (self.form.nineq,), dtype=torch.bool, device=G.device
                )
                self._G_keep_scipy = self._G_full_scipy
                return

            rc = torch.nonzero(
                nz_mask[row_ids], as_tuple=False
            )  # (k,2) -> [row_in_subset, col]
            cols = rc[:, 1]
            coeff = G[row_ids, cols]

            is_pos1 = torch.isclose(coeff, torch.ones_like(coeff))
            is_neg1 = torch.isclose(coeff, -torch.ones_like(coeff))
            is_pm1 = is_pos1 | is_neg1

            bound_row_ids = row_ids[is_pm1]
            bound_cols = cols[is_pm1]
            bound_sign = torch.where(
                is_pos1[is_pm1],
                torch.ones_like(bound_cols),
                -torch.ones_like(bound_cols),
            ).to(G.dtype)

            keep_mask = torch.ones(
                (self.form.nineq,), dtype=torch.bool, device=G.device
            )
            keep_mask[bound_row_ids] = False

            self.bound_row_ids = bound_row_ids
            self.bound_cols = bound_cols
            self.bound_sign = bound_sign
            self.keep_row_mask = keep_mask

            # Cache reduced G for SciPy
            G_keep = self.form.G[self.keep_row_mask, :]
            G_keep_cpu = G_keep.detach().cpu().numpy()
            self._G_keep_scipy = (
                sparse.csr_matrix(G_keep_cpu) if self.use_sparse else G_keep_cpu
            )

    def _bounds_from_h(
        self, h_i: torch.Tensor
    ) -> Tuple[Optional[List[Tuple[float, float]]], np.ndarray, object]:
        """
        Given h_i (nineq,) for one instance, build (bounds, h_ub, A_ub).
        Returns:
          bounds: list[(lb,ub)] or None
          h_ub: numpy RHS matching A_ub rows
          A_ub: scipy matrix for inequalities
        """
        if (not self.extract_bounds) or (self.bound_row_ids is None):
            A_ub = self._G_full_scipy
            h_ub = h_i.detach().cpu().numpy().astype(np.float64, copy=False)
            return None, h_ub, A_ub

        nz = self.form.nz
        lb = np.full((nz,), -np.inf, dtype=np.float64)
        ub = np.full((nz,), np.inf, dtype=np.float64)

        rhs = (
            h_i[self.bound_row_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        cols = self.bound_cols.detach().cpu().numpy()
        signs = self.bound_sign.detach().cpu().numpy().astype(np.float64, copy=False)

        # +1 * x_j <= u  -> ub[j] = min(ub[j], u)
        pos = signs > 0
        if np.any(pos):
            j = cols[pos]
            u = rhs[pos]
            for jj, uu in zip(j, u):
                if uu < ub[jj]:
                    ub[jj] = uu

        # -1 * x_j <= -l  <=> x_j >= l  -> lb[j] = max(lb[j], l) with l = -rhs
        neg = signs < 0
        if np.any(neg):
            j = cols[neg]
            l = -rhs[neg]
            for jj, ll in zip(j, l):
                if ll > lb[jj]:
                    lb[jj] = ll

        # remaining inequalities
        h_ub = (
            h_i[self.keep_row_mask]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        A_ub = self._G_keep_scipy
        bounds = list(zip(lb.tolist(), ub.tolist()))
        return bounds, h_ub, A_ub

    def solve(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        *,
        return_scipy_results: bool = False,
        highs_time_limit: Optional[float] = None,
    ):
        q, h, b, load, *_ = self.rhs.build(
            load, solar_max, wind_max, is_on, is_charging, is_discharging
        )
        B = load.shape[0]

        xs = []
        results = []

        for i in range(B):
            c = q[i].detach().cpu().numpy().astype(np.float64, copy=False)
            b_eq = b[i].detach().cpu().numpy().astype(np.float64, copy=False)

            bounds, h_ub, A_ub = self._bounds_from_h(h[i])

            res = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=h_ub,
                A_eq=self._A_scipy,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"time_limit": highs_time_limit}
                if highs_time_limit is not None
                else None,
            )
            results.append(res)
            if not res.success:
                raise RuntimeError(f"HiGHS LP failed (batch {i}): {res.message}")
            xs.append(
                torch.tensor(res.x, device=self.form.device, dtype=self.form.dtype)
            )

        x = torch.stack(xs, dim=0)  # (B, nz)
        return (x, results) if return_scipy_results else x


# ============================================================
# Differentiable objective via custom backward (dual-based)
# ============================================================


class EDHiGHSObjectiveFn(Function):
    """
    Differentiable LP objective value using SciPy HiGHS in forward,
    and your manual dual-based gradients in backward.

    IMPORTANT:
      - This ALWAYS solves with FULL G (no bound extraction) so that
        lam_ub aligns with form.ub_rows / row_ids and your backward scatter logic.
    """

    @staticmethod
    def forward(
        ctx,
        form,
        rhs_builder,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        highs_time_limit: Optional[float] = None,
        parallel_solve: bool = True,
        lp_workers: Optional[int] = None,
        parallel_min_batch: int = 8,
        chunks_per_worker: int = 1,
    ):
        ctx.G_th_full = is_on.shape[-1]

        (
            q,
            h,
            b,
            load,
            solar_max,
            wind_max,
            is_on,
            is_charging,
            is_discharging,
            shape_flags,
        ) = rhs_builder.build(
            load, solar_max, wind_max, is_on, is_charging, is_discharging
        )
        B = load.shape[0]

        # Cache SciPy matrices once per forward (cheap: just references)
        A_eq = sparse.csr_matrix(form.A.detach().cpu().numpy())
        A_ub = sparse.csr_matrix(form.G.detach().cpu().numpy())

        # Convert batched RHS once
        c_np = q.detach().cpu().numpy().astype(np.float64, copy=False)  # (B,nz)
        h_np = h.detach().cpu().numpy().astype(np.float64, copy=False)  # (B,nineq)
        b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)  # (B,neq)

        tasks = [
            (c_np[i], h_np[i], b_np[i], (None, None)) for i in range(B)
        ]  # bounds=None (full G)

        f_list = []
        lam_list = []
        nu_list = []

        do_parallel = (
            bool(parallel_solve)
            and B >= int(parallel_min_batch)
            and (os.cpu_count() or 1) > 1
        )

        if not do_parallel:
            for i in range(B):
                res = linprog(
                    c=tasks[i][0],
                    A_ub=A_ub,
                    b_ub=tasks[i][1],
                    A_eq=A_eq,
                    b_eq=tasks[i][2],
                    bounds=(None, None),
                    method="highs",
                    options={"time_limit": highs_time_limit}
                    if highs_time_limit is not None
                    else None,
                )
                if not res.success:
                    raise RuntimeError(f"HiGHS LP failed (batch {i}): {res.message}")
                f_list.append(float(res.fun))
                lam_list.append(res.ineqlin.marginals)
                nu_list.append(res.eqlin.marginals)
        else:
            cpu = os.cpu_count() or 1
            requested = lp_workers if lp_workers is not None else cpu
            n_jobs = min(requested, cpu, B)

            # ---- build chunks ----
            # Default: 2 chunks per worker (good load balance with low overhead)
            n_chunks = min(B, chunks_per_worker * n_jobs)
            chunk_size = (B + n_chunks - 1) // n_chunks  # ceil

            chunks = [
                list(range(s, min(B, s + chunk_size))) for s in range(0, B, chunk_size)
            ]

            outs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_solve_highs_chunk)(
                    idxs,
                    c_np,
                    h_np,
                    b_np,
                    A_ub,
                    A_eq,
                    options={"time_limit": highs_time_limit},
                )
                for idxs in chunks
            )

            # Scatter back into per-batch lists
            f_list = [None] * B
            lam_list = [None] * B
            nu_list = [None] * B

            total_compute = 0.0
            total_lp = 0.0
            total_overhead = 0.0

            for idxs, ok, msg, funs, lams, nus in outs:
                if not ok:
                    raise RuntimeError(f"HiGHS LP failed: {msg}")

                for k, i in enumerate(idxs):
                    f_list[i] = funs[k]
                    lam_list[i] = lams[k]
                    nu_list[i] = nus[k]

        device = form.device
        dtype = load.dtype

        assert all(v is not None for v in f_list)

        f = torch.as_tensor(np.array(f_list), device=device, dtype=dtype)  # (B,)
        lam_ub = torch.as_tensor(
            np.stack(lam_list), device=device, dtype=dtype
        )  # (B,nineq)
        nu_eq = torch.as_tensor(
            np.stack(nu_list), device=device, dtype=dtype
        )  # (B,neq)

        # Save for backward
        ctx.form = form
        ctx.shape_flags = shape_flags
        ctx.save_for_backward(
            lam_ub, nu_eq, load, solar_max, wind_max, is_on, is_charging, is_discharging
        )
        return f

    @staticmethod
    def backward(ctx, grad_out):
        form = ctx.form
        (
            lam_ub,
            nu_eq,
            load,
            solar_max,
            wind_max,
            is_on_lp,
            is_charging,
            is_discharging,
        ) = ctx.saved_tensors

        if grad_out.dim() == 0:
            grad_out = grad_out.unsqueeze(0)
        go = grad_out.view(-1, 1)  # (B,1)

        df_dh = lam_ub * go  # (B,nineq)
        df_db = nu_eq * go  # (B,neq)

        # Allocate grads (batched, time-first)
        g_load = torch.zeros_like(load)  # (B,T)
        g_solar = torch.zeros_like(solar_max)  # (B,T)
        g_wind = torch.zeros_like(wind_max)  # (B,T)

        # LP-space gradient for is_on (matches is_on_lp)
        g_is_on_lp = torch.zeros_like(is_on_lp)  # (B,T,G_lp)

        g_is_chg = torch.zeros_like(is_charging)  # (B,T,S)
        g_is_dis = torch.zeros_like(is_discharging)  # (B,T,S)

        # Full-thermal gradient buffer (what we must return)
        B, T = load.shape[0], load.shape[1]
        G_th_full = int(ctx.G_th_full)
        g_is_on_full = torch.zeros(
            (B, T, G_th_full), device=is_on_lp.device, dtype=is_on_lp.dtype
        )

        lp = form.lp_gen_idx  # (G_lp,) indices into FULL thermal ordering

        # ---- Inequality RHS that depend on inputs ----

        # pg_ub rows: solar/wind maxima
        for row_id in form.ub_rows.get("pg_ub", []):
            _, (t, p) = form.h_spec[row_id]
            if p == form.pg_idx_solar:
                g_solar[:, t] += df_dh[:, row_id]
            elif p == form.pg_idx_wind:
                g_wind[:, t] += df_dh[:, row_id]

        # storage charge/discharge bounds depend on is_charging/is_discharging
        for row_id in form.ub_rows.get("st_max_charge", []):
            _, (t, s) = form.h_spec[row_id]
            g_is_chg[:, t, s] += df_dh[:, row_id] * form.st_max_charge[s]

        for row_id in form.ub_rows.get("st_min_charge", []):
            _, (t, s) = form.h_spec[row_id]
            g_is_chg[:, t, s] += df_dh[:, row_id] * (-form.st_min_charge[s])

        for row_id in form.ub_rows.get("st_max_discharge", []):
            _, (t, s) = form.h_spec[row_id]
            g_is_dis[:, t, s] += df_dh[:, row_id] * form.st_max_discharge[s]

        for row_id in form.ub_rows.get("st_min_discharge", []):
            _, (t, s) = form.h_spec[row_id]
            g_is_dis[:, t, s] += df_dh[:, row_id] * (-form.st_min_discharge[s])

        # thermal upper bounds depend on is_on (LP space)
        for row_id in form.ub_rows.get("pa_ub_on", []):
            _, (t, g) = form.h_spec[row_id]  # g in [0..G_lp-1]
            g_is_on_lp[:, t, g] += df_dh[:, row_id] * form.th_power_diff[g]

        for row_id in form.ub_rows.get("seg_ub_on", []):
            _, (t, g, k) = form.h_spec[row_id]
            g_is_on_lp[:, t, g] += df_dh[:, row_id] * form.th_seg_mw[g, k]

        # curtailment ub depends on load
        for row_id in form.ub_rows.get("curt_ub", []):
            _, t = form.h_spec[row_id]
            g_load[:, t] += df_dh[:, row_id]

        # ---- Equality RHS that depend on inputs ----
        # power balance RHS: b[t] = load[t] - sum_g pmin[g]*is_on[t,g]  (LP space)
        for row_id in form.eq_rows.get("power_balance", []):
            _, t = form.b_spec[row_id]
            g_load[:, t] += df_db[:, row_id]
            g_is_on_lp[:, t, :] += df_db[:, row_id].unsqueeze(1) * (
                -form.th_min_power.view(1, -1)
            )

        # ---- Scatter LP grads back to full thermal space ----
        g_is_on_full[:, :, lp] = g_is_on_lp

        # Unbatch back to original shapes if caller passed unbatched tensors
        flags = ctx.shape_flags
        if flags["load_was_1d"]:
            g_load = g_load.squeeze(0)
        if flags["solar_was_1d"]:
            g_solar = g_solar.squeeze(0)
        if flags["wind_was_1d"]:
            g_wind = g_wind.squeeze(0)
        if flags["is_on_was_2d"]:
            g_is_on_full = g_is_on_full.squeeze(0)
        if flags["is_chg_was_2d"]:
            g_is_chg = g_is_chg.squeeze(0)
        if flags["is_dis_was_2d"]:
            g_is_dis = g_is_dis.squeeze(0)

        # None for (form, rhs_builder) and scalar args
        return (
            None,  # form
            None,  # rhs_builder
            g_load,
            g_solar,
            g_wind,
            g_is_on_full,  # FULL thermal grad (matches input is_on)
            g_is_chg,
            g_is_dis,
            None,
            None,
            None,
            None,
            None,
        )


@registry.register_ed_model("lp_scipy")
class EDModelLP(nn.Module):
    """
    Full SciPy/HiGHS “version” in the same clean architecture:

      - form: EDFormulation
      - rhs:  EDRHSBuilder
      - solver: EDHiGHSSolver           (primal x, optional bounds extraction)
      - objective(): EDHiGHSObjectiveFn (differentiable objective with custom backward)

    """

    def __init__(
        self,
        *,
        instance_path,
        device="cpu",
        dtype=torch.float32,
        extract_bounds=False,
        use_sparse=True,
        parallel_solve=True,
        lp_workers=None,
        parallel_min_batch=8,
        lp_gen_idx=None,
        chunks_per_worker=1,
    ):
        super().__init__()
        # NOTE: eps is irrelevant for HiGHS objective; keep eps=0.0 in the formulation
        ed_data_dict = create_data_dict(instance_path)
        self.lp_gen_idx = lp_gen_idx if lp_gen_idx is not None else torch.arange(51)
        self.chunks_per_worker = chunks_per_worker

        self.form = EDFormulation(
            ed_data_dict,
            eps=0.0,
            device=device,
            dtype=dtype,
            lp_gen_idx=self.lp_gen_idx,
        )

        self.rhs = EDRHSBuilder(self.form)
        self.solver = EDHiGHSSolver(
            self.form,
            self.rhs,
            extract_bounds=extract_bounds,
            use_sparse=use_sparse,
        )

        self.parallel_solve = bool(parallel_solve)
        self.lp_workers = lp_workers
        self.parallel_min_batch = int(parallel_min_batch)

    def forward(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        *,
        return_scipy_results=False,
        highs_time_limit: Optional[float] = None,
    ):
        return self.solver.solve(
            load,
            solar_max,
            wind_max,
            is_on,
            is_charging,
            is_discharging,
            return_scipy_results=return_scipy_results,
            highs_time_limit=highs_time_limit,
        )

    def objective(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        *,
        highs_time_limit: Optional[float] = None,
    ):
        return EDHiGHSObjectiveFn.apply(
            self.form,
            self.rhs,
            load,
            solar_max,
            wind_max,
            is_on,
            is_charging,
            is_discharging,
            highs_time_limit,
            self.parallel_solve,
            self.lp_workers,
            self.parallel_min_batch,
            self.chunks_per_worker,
        )
