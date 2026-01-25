import torch
import torch.nn as nn
from qpth.qp import QPFunction


class EDShapes:
    def __init__(self, G, P, S, K, T):
        self.G, self.P, self.S, self.K, self.T = int(G), int(P), int(S), int(K), int(T)

class VarIndex:
    def __init__(self, sh: EDShapes):
        self.sh = sh
        G, P, S, K, T = sh.G, sh.P, sh.S, sh.K, sh.T

        off = 0 # offset
        self.off_pg   = off; off += P * T              # profiled_generation[P,T]
        self.off_s    = off; off += S * T              # storage_level[S,T]
        self.off_cr   = off; off += S * T              # charge_rate[S,T]
        self.off_dr   = off; off += S * T              # discharge_rate[S,T]
        self.off_seg  = off; off += G * T * K          # segprod[G,T,K]
        self.off_pa   = off; off += G * T              # prod_above[G,T]
        self.off_curt = off; off += T                  # curtailment[T]
        self.nz = off # total number of variables

    def pg(self, p, t):          return self.off_pg   + p*self.sh.T + t # profiled_generation
    def s(self, u, t):           return self.off_s    + u*self.sh.T + t # storage_level
    def cr(self, u, t):          return self.off_cr   + u*self.sh.T + t # charge_rate
    def dr(self, u, t):          return self.off_dr   + u*self.sh.T + t # discharge_rate
    def seg(self, g, t, k):      return self.off_seg  + (g*self.sh.T + t)*self.sh.K + k # segprod
    def pa(self, g, t):          return self.off_pa   + g*self.sh.T + t # prod_above
    def curt(self, t):           return self.off_curt + t # curtailment

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
    def __init__(self, ed_data_dict, eps=1e-5, device="cpu"):
        super().__init__()
        self.eps = eps
        self.device = torch.device(device)
        self.T = len(ed_data_dict["system_data"].load)
        self.G = len(ed_data_dict["thermal_gen_data_list"])
        self.P = len(ed_data_dict["profiled_gen_data_list"])
        self.S = len(ed_data_dict["storage_data_list"])
        self.K = max(len(g.production_cost_curve) for g in ed_data_dict["thermal_gen_data_list"])

        self.sh = EDShapes(G=self.G, P=self.P, S=self.S, K=self.K, T=self.T)
        self.idx = VarIndex(self.sh)
        nz = self.idx.nz

        # Initialize p 
        self.p0 = torch.zeros(nz, device=self.device)
        self.register_buffer("p0", self.p0)

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

    def forward(self, load, solar_max, wind_max, is_on, is_charging, is_discharging):
        B = load.shape[0]  # batch size
        h = torch.zeros((B, self.nineq), device=self.device)
        b = torch.zeros((B, self.neq), device=self.device)
        p = self.p0.unsqueeze(0).expand(B, -1)  # (B, nz)

        self._fill_profiled_rhs(h, solar_max, wind_max)
        self._fill_storage_rhs(b, h, is_charging, is_discharging)
        self._fill_thermal_rhs(b, is_on)
        self._fill_system_rhs(b, load) 

        y = QPFunction(verbose=False)(Q, p, G, h, A, b)
        return y # you should probably have this be a dict
    
    # -- helper functions --
    
    def _add_profiled_gen_rows(self, builder: RowBuilder, ed_data_dict):
        sh = self.sh
        idx = self.idx

        profiled = sorted(ed_data_dict["profiled_gen_data_list"], key= lambda g: g.name)
        names = [g.name for g in profiled]
        min_power = torch.tensor([g.min_power for g in profiled], device=builder.device, dtype=builder.dtype)
        max_power = torch.tensor([g.max_power for g in profiled], device=builder.device, dtype=builder.dtype)

        self.pg_idx_solar = names.index("solar")
        self.pg_idx_wind = names.index("wind")

        cost = torch.tensor([g.cost for g in profiled], device=builder.device, dtype=builder.dtype)
        
        for p in range(sh.P): # TODO: think about vectorizing this later
            for t in range(sh.T):
                # Add gen cost
                self.p0[idx.pg(p, t)] = cost[p]

                # Adding pmin constraints: - pg[p, t] <= - min_power[p]
                builder.add_ub_row(
                    {idx.pg(p, t): -1.0},
                    rhs_key="pg_lb",
                    meta=(p, t)
                )

                # Adding pmax constraints: pg[p, t] <= max_power[p]
                builder.add_ub_row(
                    {idx.pg(p, t): 1.0},
                    rhs_key="pg_ub",
                    meta=(p, t)
                )
        
        self.register_buffer("pg_min_power", min_power)  # (P,)
        self.register_buffer("pg_max_power", max_power)  # (P,)

    def _add_storage_rows(self, builder: RowBuilder, ed_data_dict):
        sh = self.sh
        idx = self.idx
        storage = sorted(ed_data_dict["storage_data_list"], key=lambda s: s.name)

        max_levels = torch.tensor([s.max_level for s in storage], device=builder.device, dtype=builder.dtype)
        min_levels = torch.tensor([getattr(d, "min_level", 0.0) for d in storage], device=builder.device, dtype=builder.dtype)  
        
        min_charge = torch.tensor([getattr(d, "min_charge_rate", 0.0) for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)
        max_charge = torch.tensor([d.max_charge_rate for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)

        min_discharge = torch.tensor([getattr(d, "min_discharge_rate", 0.0) for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)
        max_discharge = torch.tensor([d.max_discharge_rate for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)

        init_levels = torch.tensor([d.initial_level for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)
        loss_factors = torch.tensor(
            [getattr(d, "loss_factor", 0.0) for d in storage],
            device=builder.device, dtype=builder.dtype
        )  # (U,)

        charge_eff = torch.tensor([d.charge_efficiency for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)
        discharge_eff = torch.tensor([d.discharge_efficiency for d in storage], device=builder.device, dtype=builder.dtype)  # (U,)

        min_end = torch.tensor(
            [getattr(d, "min_ending_level", getattr(d, "min_level", 0.0)) for d in storage],
            device=builder.device, dtype=builder.dtype
        )  # (U,)
        max_end = torch.tensor(
            [getattr(d, "max_ending_level", d.max_level) for d in storage],
            device=builder.device, dtype=builder.dtype
        )  # (U,)

        charge_costs = torch.tensor([data.charge_cost for data in storage], device=builder.device, dtype=builder.dtype)
        discharge_costs = torch.tensor([data.discharge_cost for data in storage], device=builder.device, dtype=builder.dtype)

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
                    {idx.s(s, t): 1.0},
                    rhs_key="st_max_level",
                    meta=(s, t)
                )
                builder.add_ub_row(
                    {idx.s(s, t): -1.0},
                    rhs_key="st_min_level",
                    meta=(s, t)
                )

                # Charge rate constraints
                builder.add_ub_row(
                    {idx.cr(s, t): 1.0},
                    rhs_key="st_max_charge",
                    meta=(s, t)
                )
                builder.add_ub_row(
                    {idx.cr(s, t): -1.0},
                    rhs_key="st_min_charge",
                    meta=(s, t)
                )

                # Discharge rate constraints
                builder.add_ub_row(
                    {idx.dr(s, t): 1.0},
                    rhs_key="st_max_discharge",
                    meta=(s, t)
                )
                builder.add_ub_row(
                    {idx.dr(s, t): -1.0},
                    rhs_key="st_min_discharge",
                    meta=(s, t)
                )

            # End of horizon bounds
            builder.add_ub_row(
                {idx.s(s, sh.T - 1): 1.0},
                rhs_key="st_max_end",
                meta=s
            )
            builder.add_ub_row(
                {idx.s(s, sh.T - 1): -1.0},
                rhs_key="st_min_end",
                meta=s
            )

        # Equality constraints: storage level
        for s in range(sh.S):
            for t in range(sh.T):
                coeffs = {}
                if t == 0:
                    coeffs[idx.s(s, t)] = 1.0
                    coeffs[idx.cr(s, t)] = -charge_eff[s]
                    coeffs[idx.dr(s, t)] = 1.0 / discharge_eff[s]
                    rhs_key = f"st_level_init"
                else:
                    coeffs[idx.s(s, t)] = 1.0
                    coeffs[idx.s(s, t - 1)] = -(1.0 - loss_factors[s])
                    coeffs[idx.cr(s, t)] = -charge_eff[s]
                    coeffs[idx.dr(s, t)] = 1.0 / discharge_eff[s]
                    rhs_key = f"st_level_evol_{s}"
                
                builder.add_eq_row(
                    coeffs,
                    rhs_key=rhs_key,
                    meta=(s, t)
                )
    
    def _add_thermal_rows(self, builder, ed_data_dict): # TODO: check the segment logic 
        sh = self.sh
        idx = self.idx
        thermals = sorted(ed_data_dict["thermal_gen_data_list"], key=lambda g: g.name)
        G, T, K = sh.G, sh.T, sh.K

        # ---- constants ----
        min_power = torch.tensor([g.min_power for g in thermals], device=builder.device, dtype=builder.dtype)  # (G,)
        max_power = torch.tensor([g.max_power for g in thermals], device=builder.device, dtype=builder.dtype)  # (G,)

        power_diff = torch.clamp(max_power - min_power, min=0.0)  # (G,)
        power_diff = torch.where(power_diff < 1e-7, torch.zeros_like(power_diff), power_diff)

        # Build segment_mw and segment_cost as (G,K) with zero-padding
        seg_mw = torch.zeros((G, K), device=self.device, dtype=self.dtype)
        seg_cost = torch.zeros((G, K), device=self.device, dtype=self.dtype)
        for gi, g in enumerate(thermals):
            curve = g.production_cost_curve  # list of (mw, cost)
            n = len(curve)
            seg_mw[gi, :n] = torch.tensor([mw for mw, _ in curve], device=self.device, dtype=self.dtype)
            seg_cost[gi, :n] = torch.tensor([c for _, c in curve], device=self.device, dtype=self.dtype)

        # Save as buffers for RHS fill + objective
        self.register_buffer("th_min_power", min_power)
        self.register_buffer("th_max_power", max_power)
        self.register_buffer("th_power_diff", power_diff)
        self.register_buffer("th_seg_mw", seg_mw)       # (G,K)
        self.register_buffer("th_seg_cost", seg_cost)   # (G,K)

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
                    { idx.pa(g, t): 1.0 },
                    rhs_key="pa_ub_on",
                    meta=(g, t),
                )

        # (2) segprod[g,t,k] <= seg_mw[g,k] * is_on[g,t]
        for g in range(G):
            for t in range(T):
                for k in range(K):
                    builder.add_ub_row(
                        { idx.seg(g, t, k): 1.0 },
                        rhs_key="seg_ub_on",
                        meta=(g, t, k),
                    )

        # (Optional but recommended) Nonnegativity explicitly:
        # segprod >= 0  -> -segprod <= 0
        # prod_above >= 0 -> -prod_above <= 0
        # If you already enforce via other bounds and eps-QP, still safer to include.
        for g in range(G):
            for t in range(T):
                builder.add_ub_row({ idx.pa(g, t): -1.0 }, rhs_key="pa_nn", meta=(g, t))
                for k in range(K):
                    builder.add_ub_row({ idx.seg(g, t, k): -1.0 }, rhs_key="seg_nn", meta=(g, t, k))

        # -------------------------
        # Equalities: A z = b
        # -------------------------

        # (4) prod_above[g,t] == sum_k segprod[g,t,k]
        # -> prod_above[g,t] - sum_k segprod[g,t,k] = 0
        for g in range(G):
            for t in range(T):
                coeffs = { idx.pa(g, t): 1.0 }
                for k in range(K):
                    coeffs[idx.seg(g, t, k)] = -1.0
                builder.add_eq_row(coeffs, rhs_key="pa_link", meta=(g, t))

    def _add_system_rows(self, builder, ed_data_dict):
        sh = self.sh
        idx = self.idx

        power_balance_penalty = ed_data_dict["system_data"].power_balance_penalty


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

            builder.add_eq_row(
                coeffs,
                rhs_key="power_balance",
                meta=t
            )

    def _fill_profiled_rhs(self, h, solar_max, wind_max):

        # pg lower bounds
        for row_id in self.builder.ub_rows.get("pg_lb", []):
            rhs_key, meta = self.h_spec[row_id]
            p, t = meta
            h[:, row_id] = -self.pg_min_power[p]

        # pg upper bounds
        for row_id in self.builder.ub_rows.get("pg_ub", []):
            rhs_key, meta = self.h_spec[row_id]
            p, t = meta
            if p == self.pg_idx_solar:
                h[:, row_id] = solar_max[:, t]
            elif p == self.pg_idx_wind:
                h[:, row_id] = wind_max[:, t]
            else:
                h[:, row_id] = self.pg_max_power[p]

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
    
    def _fill_system_rhs(self, b, load):
        # Power balance
        for row_id in self.builder.eq_rows.get("power_balance", []):
            _, t = self.b_spec[row_id]
            b[:, row_id] = load[:, t]