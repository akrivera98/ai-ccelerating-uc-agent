from dataclasses import dataclass
import json
from typing import Tuple


# Component Data
@dataclass
class ProfiledGeneratorData:
    name: str
    cost: float
    min_power: list[float]
    max_power: list[float]


@dataclass
class ThermalGeneratorData:
    name: str
    production_cost_curve: list[Tuple[float, float]]  # List of ($, MW) pairs
    startup_costs: list[float]
    startup_delays: list[float]  # or int # double-check how this will be used
    ramp_up_limit: float
    ramp_down_limit: float
    startup_limit: float
    shutdown_limit: float
    min_down_time: float
    min_up_time: float
    initial_power: float
    initial_status: float
    min_power: float
    max_power: float


@dataclass
class SystemData:
    power_balance_penalty: float
    load: list[float]


@dataclass
class StorageUnitData:
    name: str
    max_level: float
    charge_cost: float
    discharge_cost: float
    charge_efficiency: float
    discharge_efficiency: float
    max_charge_rate: float
    max_discharge_rate: float
    initial_level: float
    initial_level: float
    min_ending_level: float


def create_data_dict(instance_path: str) -> dict:
    # Read json file from data instance
    with open(instance_path, "r") as f:
        instance_data = json.load(f)

    profiled_gens_data_list = []
    storage_units_data_list = []
    thermal_gens_data_list = []

    for g_name, g_data in instance_data["Generators"].items():
        if g_data["Type"] == "Profiled":
            profiled_gens_data_list.append(_parse_profiled_gen_data(g_name, g_data))
        elif g_data["Type"] == "Thermal":
            thermal_gens_data_list.append(_parse_thermal_gen_data(g_name, g_data))

    for s_name, s_data in instance_data["Storage units"].items():
        storage_units_data_list.append(_parse_storage_units_data(s_name, s_data))

    system_data = SystemData(
        power_balance_penalty=instance_data["Parameters"][
            "Power balance penalty ($/MW)"
        ],
        load=instance_data["Buses"]["b1"]["Load (MW)"], # should be a parameter, does it matter if i have it here too?
    )

    data_dict = {
        "profiled_gen_data_list": profiled_gens_data_list,
        "storage_data_list": storage_units_data_list,
        "thermal_gen_data_list": thermal_gens_data_list,
        "system_data": system_data,
    }
    return data_dict


def _parse_profiled_gen_data(gen_name: str, gen_data: dict) -> ProfiledGeneratorData:
    min_power = gen_data["Minimum power (MW)"]
    max_power = gen_data["Maximum power (MW)"]
    if isinstance(min_power, (int, float)):
        min_power = [min_power] * 72
    if isinstance(max_power, (int, float)):
        max_power = [max_power] * 72

    return ProfiledGeneratorData(
        name=gen_name,
        cost=gen_data["Cost ($/MW)"],
        min_power=min_power,
        max_power=max_power,
    )


def _parse_storage_units_data(storage_name: str, storage_data: dict) -> StorageUnitData:
    return StorageUnitData(
        name=storage_name,
        max_level=storage_data["Maximum level (MWh)"],
        charge_cost=storage_data["Charge cost ($/MW)"],
        discharge_cost=storage_data["Discharge cost ($/MW)"],
        charge_efficiency=storage_data["Charge efficiency"],
        discharge_efficiency=storage_data["Discharge efficiency"],
        max_charge_rate=storage_data["Maximum charge rate (MW)"],
        max_discharge_rate=storage_data["Maximum discharge rate (MW)"],
        initial_level=storage_data["Initial level (MWh)"],
        min_ending_level=storage_data["Last period minimum level (MWh)"],
    )


def _parse_thermal_gen_data(gen_name: str, gen_data: dict) -> ThermalGeneratorData:
    return ThermalGeneratorData(
        name=gen_name,
        production_cost_curve=_parse_production_cost_curve(
            gen_data["Production cost curve (MW)"],
            gen_data["Production cost curve ($)"],
        ),
        startup_costs=[gen_data["Startup costs ($)"][0]],
        startup_delays=[gen_data["Startup delays (h)"][0]],
        ramp_up_limit=gen_data["Ramp up limit (MW)"],
        ramp_down_limit=gen_data["Ramp down limit (MW)"],
        startup_limit=gen_data["Startup limit (MW)"],  # not used currently
        shutdown_limit=gen_data["Shutdown limit (MW)"],  # not used currently
        min_down_time=gen_data["Minimum downtime (h)"],
        min_up_time=gen_data["Minimum uptime (h)"],
        initial_power=gen_data["Initial power (MW)"],
        initial_status=gen_data["Initial status (h)"],
        min_power=gen_data["Production cost curve (MW)"][0],
        max_power=gen_data["Production cost curve (MW)"][-1],
    )


def _parse_production_cost_curve(mw_list, cost_list) -> list[Tuple[float, float]]:
    og_curve = []
    final_curve = []
    og_curve = list(zip(mw_list, cost_list))

    num_segments = len(og_curve)
    for k in range(1, num_segments):
        p1 = og_curve[k - 1][0]
        p2 = og_curve[k][0]
        c1 = og_curve[k - 1][1]
        c2 = og_curve[k][1]
        marginal_cost = (c2 - c1) / (p2 - p1)
        final_curve.append((p2 - p1, marginal_cost))

    return final_curve
