import torch
import torch.nn as nn


class EDShapes:
    def __init__(self, G, P, S, K, T):
        self.G, self.P, self.S, self.K, self.T = int(G), int(P), int(S), int(K), int(T)


class VarIndex:
    """
    Time-major variable indexing.

      - pg:   (T, P) # profiled gen production
      - s:    (T, S) # storage level
      - cr:   (T, S) # charge rate
      - dr:   (T, S) # discharge rate
      - seg:  (T, G, K) # production in segment k (linked to pa)
      - pa:   (T, G) # production above minimum (linked to seg)
      - curt: (T,) # curtailment
    """

    def __init__(self, sh: EDShapes):
        self.sh = sh
        G, P, S, K, T = sh.G, sh.P, sh.S, sh.K, sh.T

        off = 0
        self.off_pg = off
        off += T * P  # pg[T,P]

        self.off_s = off
        off += T * S  # s[T,S]

        self.off_cr = off
        off += T * S  # cr[T,S]

        self.off_dr = off
        off += T * S  # dr[T,S]

        self.off_seg = off
        off += T * G * K  # seg[T,G,K]

        self.off_pa = off
        off += T * G  # pa[T,G]

        self.off_curt = off
        off += T  # curt[T]

        self.nz = off

    def pg(self, t, p):
        return self.off_pg + t * self.sh.P + p

    def s(self, t, u):
        return self.off_s + t * self.sh.S + u

    def cr(self, t, u):
        return self.off_cr + t * self.sh.S + u

    def dr(self, t, u):
        return self.off_dr + t * self.sh.S + u

    def seg(self, t, g, k):
        return self.off_seg + t * (self.sh.G * self.sh.K) + g * self.sh.K + k

    def pa(self, t, g):
        return self.off_pa + t * self.sh.G + g

    def curt(self, t):
        return self.off_curt + t


class RowBuilder:
    """
    Dense row builder.

    Collects sparse row descriptions, then materializes dense A and G matrices.
    Also records (rhs_key, meta) per row so RHS can be filled later without rebuilding A/G.
    """

    def __init__(self, nz, device="cpu", dtype=torch.float32):
        self.nz = int(nz)
        self.device = torch.device(device)
        self.dtype = dtype

        self.A_rows = []  # (idxs, vals, rhs_key, meta)
        self.G_rows = []  # (idxs, vals, rhs_key, meta)

        self.eq_rows = {}  # rhs_key -> list[row_id]
        self.ub_rows = {}  # rhs_key -> list[row_id]

    def add_eq_row(self, coeffs: dict, rhs_key: str, meta=None):
        row_id = len(self.A_rows)
        idxs = torch.tensor(list(coeffs.keys()), device=self.device, dtype=torch.long)
        vals = torch.tensor(list(coeffs.values()), device=self.device, dtype=self.dtype)
        self.A_rows.append((idxs, vals, rhs_key, meta))
        self.eq_rows.setdefault(rhs_key, []).append(row_id)
        return row_id

    def add_ub_row(self, coeffs: dict, rhs_key: str, meta=None):
        row_id = len(self.G_rows)
        idxs = torch.tensor(list(coeffs.keys()), device=self.device, dtype=torch.long)
        vals = torch.tensor(list(coeffs.values()), device=self.device, dtype=self.dtype)
        self.G_rows.append((idxs, vals, rhs_key, meta))
        self.ub_rows.setdefault(rhs_key, []).append(row_id)
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

        b_spec = [(rhs_key, meta) for (_, _, rhs_key, meta) in self.A_rows]
        h_spec = [(rhs_key, meta) for (_, _, rhs_key, meta) in self.G_rows]
        return A, G, b_spec, h_spec


class EDFormulation(nn.Module):
    """
    Solver-agnostic ED formulation template (time-major indexing).

    Owns ONLY objects that depend on ed_data_dict:
      - indexing (sh, idx, nz)
      - objective template (q_base, Q_base)
      - constant constraint matrices (A, G) and row metadata (b_spec, h_spec, row groups)
      - all static parameter buffers used later for RHS fill and/or custom backward

    It does NOT:
      - build RHS vectors (b, h) for a given instance/batch
      - solve (qpth/HiGHS/etc.)
      - do any multiprocessing / bound extraction
    """

    def __init__(
        self,
        ed_data_dict,
        *,
        eps=1e-3,
        device="cpu",
        dtype=torch.float32,
        lp_gen_idx=list(range(51)),
    ):
        super().__init__()
        self.eps = float(eps)
        self.device = torch.device(device)
        self.dtype = dtype
        self.lp_gen_idx = torch.tensor(lp_gen_idx, dtype=torch.long)

        # ---- shapes / indexing ----
        T = len(ed_data_dict["system_data"].load)
        G = len(lp_gen_idx)
        P = len(ed_data_dict["profiled_gen_data_list"])
        S = len(ed_data_dict["storage_data_list"])
        K = max(
            len(g.production_cost_curve) for g in ed_data_dict["thermal_gen_data_list"]
        )

        self.sh = EDShapes(G=G, P=P, S=S, K=K, T=T)
        self.idx = VarIndex(self.sh)
        self.nz = int(self.idx.nz)

        # ---- objective template ----
        q_base = torch.zeros(self.nz, device=self.device, dtype=self.dtype)
        self.register_buffer("q_base", q_base)

        if self.eps != 0.0:
            Q_base = self.eps * torch.eye(self.nz, device=self.device, dtype=self.dtype)
        else:
            Q_base = torch.zeros(
                (self.nz, self.nz), device=self.device, dtype=self.dtype
            )
        self.register_buffer("Q_base", Q_base)

        # ---- build canonical constraints A x = b, G x <= h ----
        builder = RowBuilder(nz=self.nz, device=self.device, dtype=self.dtype)

        self._add_profiled_gen_rows(builder, ed_data_dict)
        self._add_storage_rows(builder, ed_data_dict)
        self._add_thermal_rows(builder, ed_data_dict)
        self._add_system_rows(builder, ed_data_dict)

        A, Gm, b_spec, h_spec = builder.finalize()

        self.neq = int(A.shape[0])
        self.nineq = int(Gm.shape[0])

        self.register_buffer("A", A)
        self.register_buffer("G", Gm)

        # ---- row metadata (python objects) ----
        self.b_spec = b_spec
        self.h_spec = h_spec
        self.eq_rows = builder.eq_rows
        self.ub_rows = builder.ub_rows

        # ---- optional sanity checks for time-major layout ----
        # pg block: next asset increments by 1, next time increments by P
        if self.sh.P > 1 and self.sh.T > 1:
            assert self.idx.pg(0, 1) - self.idx.pg(0, 0) == 1
            assert self.idx.pg(1, 0) - self.idx.pg(0, 0) == self.sh.P

    # ------------------------------------------------------------------
    # -------------------- formulation construction helpers -------------
    # ------------------------------------------------------------------

    def _add_profiled_gen_rows(self, builder: RowBuilder, ed_data_dict):
        sh, idx = self.sh, self.idx

        profiled = sorted(ed_data_dict["profiled_gen_data_list"], key=lambda g: g.name)
        names = [g.name for g in profiled]
        self.profiled_units_names = names

        min_power = torch.tensor(
            [g.min_power[0] for g in profiled],
            device=builder.device,
            dtype=builder.dtype,
        )
        max_power = torch.tensor(
            [g.max_power[0] for g in profiled],
            device=builder.device,
            dtype=builder.dtype,
        )

        self.pg_idx_solar = names.index("solar")
        self.pg_idx_wind = names.index("wind")

        cost = torch.tensor(
            [g.cost for g in profiled], device=builder.device, dtype=builder.dtype
        )

        for t in range(sh.T):
            for p in range(sh.P):
                # objective: q^T x
                self.q_base[idx.pg(t, p)] = cost[p]

                # -pg[t,p] <= -pmin[p]
                builder.add_ub_row({idx.pg(t, p): -1.0}, rhs_key="pg_lb", meta=(t, p))
                # +pg[t,p] <= pmax[p] (solar/wind override RHS later)
                builder.add_ub_row({idx.pg(t, p): 1.0}, rhs_key="pg_ub", meta=(t, p))

        self.register_buffer("pg_min_power", min_power)  # (P)
        self.register_buffer("pg_max_power", max_power)  # (P)

    def _add_storage_rows(self, builder: RowBuilder, ed_data_dict):
        sh, idx = self.sh, self.idx

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
        )
        max_charge = torch.tensor(
            [d.max_charge_rate for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

        min_discharge = torch.tensor(
            [getattr(d, "min_discharge_rate", 0.0) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )
        max_discharge = torch.tensor(
            [d.max_discharge_rate for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

        init_levels = torch.tensor(
            [d.initial_level for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )
        loss_factors = torch.tensor(
            [getattr(d, "loss_factor", 0.0) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

        charge_eff = torch.tensor(
            [d.charge_efficiency for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )
        discharge_eff = torch.tensor(
            [d.discharge_efficiency for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

        min_end = torch.tensor(
            [
                getattr(d, "min_ending_level", getattr(d, "min_level", 0.0))
                for d in storage
            ],
            device=builder.device,
            dtype=builder.dtype,
        )
        max_end = torch.tensor(
            [getattr(d, "max_ending_level", d.max_level) for d in storage],
            device=builder.device,
            dtype=builder.dtype,
        )

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

        # save static parameters
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

        # inequalities + objective
        for t in range(sh.T):
            for s in range(sh.S):
                self.q_base[idx.cr(t, s)] = charge_costs[s]
                self.q_base[idx.dr(t, s)] = discharge_costs[s]

                # level bounds
                builder.add_ub_row(
                    {idx.s(t, s): 1.0}, rhs_key="st_max_level", meta=(t, s)
                )
                builder.add_ub_row(
                    {idx.s(t, s): -1.0}, rhs_key="st_min_level", meta=(t, s)
                )

                # charge bounds
                builder.add_ub_row(
                    {idx.cr(t, s): 1.0}, rhs_key="st_max_charge", meta=(t, s)
                )
                builder.add_ub_row(
                    {idx.cr(t, s): -1.0}, rhs_key="st_min_charge", meta=(t, s)
                )

                # discharge bounds
                builder.add_ub_row(
                    {idx.dr(t, s): 1.0}, rhs_key="st_max_discharge", meta=(t, s)
                )
                builder.add_ub_row(
                    {idx.dr(t, s): -1.0}, rhs_key="st_min_discharge", meta=(t, s)
                )

        # end-of-horizon bounds (t = T-1)
        t_last = sh.T - 1
        for s in range(sh.S):
            builder.add_ub_row({idx.s(t_last, s): 1.0}, rhs_key="st_max_end", meta=s)
            builder.add_ub_row({idx.s(t_last, s): -1.0}, rhs_key="st_min_end", meta=s)

        # equalities: storage evolution
        for s in range(sh.S):
            for t in range(sh.T):
                if t == 0:
                    # s[0,s] - eff_c[s]*cr[0,s] + (1/eff_d[s])*dr[0,s] = init_level[s]
                    coeffs = {
                        idx.s(t, s): 1.0,
                        idx.cr(t, s): -charge_eff[s],
                        idx.dr(t, s): 1.0 / discharge_eff[s],
                    }
                    rhs_key = "st_level_init"
                else:
                    # s[t,s] - (1-loss[s])*s[t-1,s] - eff_c[s]*cr[t,s] + (1/eff_d[s])*dr[t,s] = 0
                    coeffs = {
                        idx.s(t, s): 1.0,
                        idx.s(t - 1, s): -(1.0 - loss_factors[s]),
                        idx.cr(t, s): -charge_eff[s],
                        idx.dr(t, s): 1.0 / discharge_eff[s],
                    }
                    rhs_key = "st_level_evol"

                builder.add_eq_row(coeffs, rhs_key=rhs_key, meta=(t, s))

    def _add_thermal_rows(self, builder: RowBuilder, ed_data_dict):
        sh, idx = self.sh, self.idx

        # full thermal list in canonical order for thermals
        thermals_all = sorted(
            ed_data_dict["thermal_gen_data_list"], key=lambda g: g.name
        )

        # ---- reduce thermals using lp_gen_idx ----
        lp = self.lp_gen_idx  # LongTensor of indices into thermals_all
        thermals = [thermals_all[i] for i in lp.tolist()]  # reduced thermal list

        # IMPORTANT: from this point on, G is the reduced thermal count
        T, K = sh.T, sh.K
        G = len(thermals)

        self.thermal_units_names = [g.name for g in thermals]

        # min/max power aligned to reduced thermal list
        min_power = torch.tensor(
            [g.min_power for g in thermals], device=builder.device, dtype=builder.dtype
        )  # (G,)

        max_power = torch.tensor(
            [g.max_power for g in thermals], device=builder.device, dtype=builder.dtype
        )  # (G,)

        power_diff = torch.clamp(max_power - min_power, min=0.0)
        power_diff = torch.where(
            power_diff < 1e-7, torch.zeros_like(power_diff), power_diff
        )

        # segment data aligned to reduced thermal list
        seg_mw = torch.zeros((G, K), device=builder.device, dtype=builder.dtype)
        seg_cost = torch.zeros((G, K), device=builder.device, dtype=builder.dtype)

        for gi, g in enumerate(thermals):
            curve = g.production_cost_curve  # list of (mw, cost)
            n = min(len(curve), K)
            if n > 0:
                seg_mw[gi, :n] = torch.tensor(
                    [mw for mw, _ in curve[:n]],
                    device=builder.device,
                    dtype=builder.dtype,
                )
                seg_cost[gi, :n] = torch.tensor(
                    [c for _, c in curve[:n]],
                    device=builder.device,
                    dtype=builder.dtype,
                )

        # store reduced tensors
        self.register_buffer("th_min_power", min_power)
        self.register_buffer("th_max_power", max_power)
        self.register_buffer("th_power_diff", power_diff)
        self.register_buffer("th_seg_mw", seg_mw)
        self.register_buffer("th_seg_cost", seg_cost)

        # objective: sum seg_cost[g,k] * segprod[t,g,k]
        for t in range(T):
            for g in range(G):
                for k in range(K):
                    self.q_base[idx.seg(t, g, k)] += seg_cost[g, k]

        # inequalities: pa[t,g] <= power_diff[g] * is_on[t,g]  (rhs filled later)
        for t in range(T):
            for g in range(G):
                builder.add_ub_row({idx.pa(t, g): 1.0}, rhs_key="pa_ub_on", meta=(t, g))

        # inequalities: seg[t,g,k] <= seg_mw[g,k] * is_on[t,g] (rhs filled later)
        for t in range(T):
            for g in range(G):
                for k in range(K):
                    builder.add_ub_row(
                        {idx.seg(t, g, k): 1.0}, rhs_key="seg_ub_on", meta=(t, g, k)
                    )

        # nonnegativity: -pa <= 0, -seg <= 0
        for t in range(T):
            for g in range(G):
                builder.add_ub_row({idx.pa(t, g): -1.0}, rhs_key="pa_nn", meta=(t, g))
                for k in range(K):
                    builder.add_ub_row(
                        {idx.seg(t, g, k): -1.0}, rhs_key="seg_nn", meta=(t, g, k)
                    )

        # equality: pa[t,g] - sum_k seg[t,g,k] = 0
        for t in range(T):
            for g in range(G):
                coeffs = {idx.pa(t, g): 1.0}
                for k in range(K):
                    coeffs[idx.seg(t, g, k)] = -1.0
                builder.add_eq_row(coeffs, rhs_key="pa_link", meta=(t, g))

    def _add_system_rows(self, builder: RowBuilder, ed_data_dict):
        sh, idx = self.sh, self.idx
        power_balance_penalty = float(ed_data_dict["system_data"].power_balance_penalty)

        self.register_buffer(
            "power_balance_penalty",
            torch.tensor(
                power_balance_penalty, device=builder.device, dtype=builder.dtype
            ),
        )

        # power balance at time t:
        #   sum_p pg[t,p] + sum_g pa[t,g] + sum_s dr[t,s] - sum_s cr[t,s] + curt[t] = RHS
        for t in range(sh.T):
            # curt cost
            self.q_base[idx.curt(t)] = self.power_balance_penalty

            coeffs = {}
            for p in range(sh.P):
                coeffs[idx.pg(t, p)] = 1.0
            for g in range(sh.G):
                coeffs[idx.pa(t, g)] = 1.0
            for s in range(sh.S):
                coeffs[idx.dr(t, s)] = 1.0
            for s in range(sh.S):
                coeffs[idx.cr(t, s)] = -1.0
            coeffs[idx.curt(t)] = 1.0

            builder.add_eq_row(coeffs, rhs_key="power_balance", meta=t)

        # curt[t] <= load[t], and curt[t] >= 0 (encoded as -curt[t] <= 0)
        for t in range(sh.T):
            builder.add_ub_row({idx.curt(t): 1.0}, rhs_key="curt_ub", meta=t)
            builder.add_ub_row({idx.curt(t): -1.0}, rhs_key="curt_nn", meta=t)


class EDRHSBuilder(nn.Module):
    """
    RHS builder (batched) for a compiled EDFormulation.

    Given runtime inputs (load, solar_max, wind_max, is_on, is_charging, is_discharging),
    builds the batched vectors:
        q: (B, nz)
        h: (B, nineq)
        b: (B, neq)

      load:          (T,) or (B, T)
      solar_max:     (T,) or (B, T)
      wind_max:      (T,) or (B, T)
      is_on:         (T, G) or (B, T, G)
      is_charging:   (T, S) or (B, T, S)
      is_discharging:(T, S) or (B, T, S)
    """

    def __init__(self, form, *, device=None, dtype=None):
        super().__init__()
        self.form = form  # EDFormulation (nn.Module)
        self.device = form.device if device is None else torch.device(device)
        self.dtype = form.dtype if dtype is None else dtype

    # -------------------------
    # Public entry point
    # -------------------------
    def build(self, load, solar_max, wind_max, is_on, is_charging, is_discharging):
        """
        Returns:
            q: (B, nz)
            h: (B, nineq)
            b: (B, neq)
        """
        (
            load,
            solar_max,
            wind_max,
            is_on,
            is_charging,
            is_discharging,
            shape_flags,
        ) = self._make_batched(
            load, solar_max, wind_max, is_on, is_charging, is_discharging
        )

        B = load.shape[0]
        form = self.form
        is_on = is_on[:, :, form.lp_gen_idx]  # (B,T,G) -> (B,T,|LP|)

        # Allocate RHS
        h = torch.zeros((B, form.nineq), device=self.device, dtype=self.dtype)
        b = torch.zeros((B, form.neq), device=self.device, dtype=self.dtype)

        # Objective vector: usually constant across batch
        q = form.q_base.unsqueeze(0).expand(B, -1)

        # Fill pieces
        self._fill_profiled_rhs(h, solar_max, wind_max)
        self._fill_storage_rhs(b, h, is_charging, is_discharging)
        self._fill_thermal_rhs(b, h, is_on)
        self._fill_system_rhs(b, h, load, is_on)

        return (
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
        )

    # -------------------------
    # Batching / shape handling
    # -------------------------
    def _make_batched(
        self, load, solar_max, wind_max, is_on, is_charging, is_discharging
    ):
        """
        Normalizes inputs to:
          load, solar_max, wind_max: (B,T)
          is_on: (B,T,G)
          is_charging/is_discharging: (B,T,S)

        Returns shape flags to allow a later wrapper to unbatch if desired.
        """
        form = self.form
        T, G, S = form.sh.T, form.sh.G, form.sh.S

        # Track original shapes (for optional unbatching later)
        flags = {
            "load_was_1d": load.dim() == 1,
            "solar_was_1d": solar_max.dim() == 1,
            "wind_was_1d": wind_max.dim() == 1,
            "is_on_was_2d": is_on.dim() == 2,
            "is_chg_was_2d": is_charging.dim() == 2,
            "is_dis_was_2d": is_discharging.dim() == 2,
        }

        # Move to device/dtype (don’t detach)
        load = load.to(device=self.device, dtype=self.dtype)
        solar_max = solar_max.to(device=self.device, dtype=self.dtype)
        wind_max = wind_max.to(device=self.device, dtype=self.dtype)
        is_on = is_on.to(device=self.device, dtype=self.dtype)
        is_charging = is_charging.to(device=self.device, dtype=self.dtype)
        is_discharging = is_discharging.to(device=self.device, dtype=self.dtype)

        # Determine batch size
        if load.dim() == 1:
            B = 1
        else:
            B = load.shape[0]

        # Expand 1D -> (B,T)
        if load.dim() == 1:
            assert load.shape[0] == T
            load = load.unsqueeze(0).expand(B, -1)

        if solar_max.dim() == 1:
            assert solar_max.shape[0] == T
            solar_max = solar_max.unsqueeze(0).expand(B, -1)

        if wind_max.dim() == 1:
            assert wind_max.shape[0] == T
            wind_max = wind_max.unsqueeze(0).expand(B, -1)

        # Expand 2D -> (B,T,·)
        if is_on.dim() == 2:
            assert is_on.shape == (T, G)
            is_on = is_on.unsqueeze(0).expand(B, -1, -1)
        else:
            assert is_on.shape[1:] == (T, G)

        if is_charging.dim() == 2:
            assert is_charging.shape == (T, S)
            is_charging = is_charging.unsqueeze(0).expand(B, -1, -1)
        else:
            assert is_charging.shape[1:] == (T, S)

        if is_discharging.dim() == 2:
            assert is_discharging.shape == (T, S)
            is_discharging = is_discharging.unsqueeze(0).expand(B, -1, -1)
        else:
            assert is_discharging.shape[1:] == (T, S)

        return load, solar_max, wind_max, is_on, is_charging, is_discharging, flags

    # -------------------------
    # RHS fill helpers
    # -------------------------
    def _fill_profiled_rhs(self, h, solar_max, wind_max):
        """
        Profiled gen bounds:
          -pg[t,p] <= -pmin[p]
           pg[t,p] <= pmax[p] or solar/wind max at time t
        """
        form = self.form

        # Lower bounds: -pg <= -min_power
        for row_id in form.ub_rows.get("pg_lb", []):
            _, meta = form.h_spec[row_id]  # meta=(t,p)
            t, p = meta
            h[:, row_id] = -form.pg_min_power[p]

        # Upper bounds: pg <= max (solar/wind are time-varying inputs)
        for row_id in form.ub_rows.get("pg_ub", []):
            _, meta = form.h_spec[row_id]  # meta=(t,p)
            t, p = meta
            if p == form.pg_idx_solar:
                h[:, row_id] = solar_max[:, t]
            elif p == form.pg_idx_wind:
                h[:, row_id] = wind_max[:, t]
            else:
                h[:, row_id] = form.pg_max_power[p]

    def _fill_storage_rhs(self, b, h, is_charging, is_discharging):
        """
        Storage bounds and evolution RHS.
        Note: is_charging/is_discharging are time-first: (B,T,S)
        """
        form = self.form

        # Inequalities
        for row_id in form.ub_rows.get("st_max_level", []):
            _, (t, s) = form.h_spec[row_id]
            h[:, row_id] = form.st_max_levels[s]

        for row_id in form.ub_rows.get("st_min_level", []):
            _, (t, s) = form.h_spec[row_id]
            h[:, row_id] = -form.st_min_levels[s]

        for row_id in form.ub_rows.get("st_max_charge", []):
            _, (t, s) = form.h_spec[row_id]
            h[:, row_id] = form.st_max_charge[s] * is_charging[:, t, s]

        for row_id in form.ub_rows.get("st_min_charge", []):
            _, (t, s) = form.h_spec[row_id]
            h[:, row_id] = -form.st_min_charge[s] * is_charging[:, t, s]

        for row_id in form.ub_rows.get("st_max_discharge", []):
            _, (t, s) = form.h_spec[row_id]
            h[:, row_id] = form.st_max_discharge[s] * is_discharging[:, t, s]

        for row_id in form.ub_rows.get("st_min_discharge", []):
            _, (t, s) = form.h_spec[row_id]
            h[:, row_id] = -form.st_min_discharge[s] * is_discharging[:, t, s]

        # End-of-horizon bounds use meta=s (as set in the formulation)
        for row_id in form.ub_rows.get("st_max_end", []):
            _, s = form.h_spec[row_id]
            h[:, row_id] = form.st_max_end[s]

        for row_id in form.ub_rows.get("st_min_end", []):
            _, s = form.h_spec[row_id]
            h[:, row_id] = -form.st_min_end[s]

        # Equalities
        for row_id in form.eq_rows.get("st_level_init", []):
            _, (t, s) = form.b_spec[row_id]  # meta=(t,s) where t==0
            b[:, row_id] = form.st_init_levels[s]

        for row_id in form.eq_rows.get("st_level_evol", []):
            b[:, row_id] = 0.0

    def _fill_thermal_rhs(self, b, h, is_on):
        """
        Thermal bounds:
          pa[t,g] <= power_diff[g] * is_on[t,g]
          seg[t,g,k] <= seg_mw[g,k] * is_on[t,g]
          nonneg rows are 0
          link equality rows RHS are 0
        """
        form = self.form

        for row_id in form.ub_rows.get("pa_ub_on", []):
            _, (t, g) = form.h_spec[row_id]
            h[:, row_id] = form.th_power_diff[g] * is_on[:, t, g]

        for row_id in form.ub_rows.get("seg_ub_on", []):
            _, (t, g, k) = form.h_spec[row_id]
            h[:, row_id] = form.th_seg_mw[g, k] * is_on[:, t, g]

        for row_id in form.ub_rows.get("pa_nn", []):
            h[:, row_id] = 0.0

        for row_id in form.ub_rows.get("seg_nn", []):
            h[:, row_id] = 0.0

        for row_id in form.eq_rows.get("pa_link", []):
            b[:, row_id] = 0.0

    def _fill_system_rhs(self, b, h, load, is_on):
        """
        Power balance equality at each t has RHS:
            load[t] - sum_g min_power[g] * is_on[t,g]
        Curtailment ub: curt[t] <= load[t]
        """
        form = self.form

        for row_id in form.eq_rows.get("power_balance", []):
            _, t = form.b_spec[row_id]  # meta=t
            # pmin_on: (B,)
            pmin_on = (form.th_min_power.view(1, -1) * is_on[:, t, :]).sum(dim=1)
            b[:, row_id] = load[:, t] - pmin_on

        for row_id in form.ub_rows.get("curt_ub", []):
            _, t = form.h_spec[row_id]
            h[:, row_id] = load[:, t]

        for row_id in form.ub_rows.get("curt_nn", []):
            h[:, row_id] = 0.0
