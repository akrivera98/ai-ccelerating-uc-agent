import json
import os
import gzip
import re
import numpy as np


def get_input_data(reference_dir):
    data_in = {}
    data_out = {}
    all_files = sorted(os.listdir(reference_dir))
    list_ins = [file for file in all_files if not file.startswith(".")]
    list_ins = sorted(list_ins, key=order_key)
    for instance in list_ins:
        # with open(reference_dir + instance + "/metadata_i.json", "r") as f:
        #     data_index[index] = json.load(f)["i_done"]
        with gzip.open(reference_dir + f"/{instance}/InputData.json.gz", "r") as f:
            in_json = f.read().decode("utf-8")
            data_in[instance] = json.loads(in_json)

        with gzip.open(reference_dir + f"/{instance}/OutputData.json.gz", "r") as f:
            out_json = f.read().decode("utf-8")
            data_out[instance] = json.loads(out_json)
    return data_in, data_out


def order_key(s):
    # numb = s.split('_')
    m = re.search(r"_(\d+)$", s)

    # return int(numb[1])
    return int(m.group(1))


def diff_instance(d1, d2, path=None):
    if path is None:
        path = []

    diffs = []

    all_keys = set(d1.keys()).union(d2.keys())

    for k in all_keys:
        p = path + [k]

        if k not in d1:
            diffs.append(f"Key {'.'.join(map(str, p))} only in d2")
        elif k not in d2:
            diffs.append(f"Key {'.'.join(map(str, p))} only in d1")
        else:
            v1, v2 = d1[k], d2[k]

            if isinstance(v1, dict) and isinstance(v2, dict):
                diffs.extend(diff_instance(v1, v2, path=p))
            elif v1 != v2:
                diffs.append(f"Value mismatch at {'.'.join(map(str, p))}: {v1} != {v2}")

    return diffs


def input_data_summary(instance):
    n_gens = len(instance["Generators"])
    n_storage = len(instance["Storage units"])
    n_thermal, n_profiled = 0, 0
    for gen in instance["Generators"].values():
        if gen["Type"] == "Thermal":
            n_thermal = n_thermal + 1
        elif gen["Type"] == "Profiled":
            n_profiled = n_profiled + 1

    assert n_gens == n_thermal + n_profiled, (
        f"Generator count mismatch! Got n_gens={n_gens}, "
        f"expected {n_thermal + n_profiled} "
        "(2 extra for wind and solar aggregators)."
    )
    return {
        "n_gens": n_gens,
        "n_storage": n_storage,
        "n_thermal": n_thermal,
        "n_profiled": n_profiled,
    }


def check_generation_summation(instance):
    wind = np.array(instance["Generators"]["wind"]["Maximum power (MW)"])
    solar = np.array(instance["Generators"]["solar"]["Maximum power (MW)"])

    gen_sum = np.zeros_like(wind)
    for gen_name, gen in instance["Generators"].items():
        if gen_name not in ["solar", "wind"]:
            if gen["Type"] == "Profiled":
                gen_sum += np.array(gen["Maximum power (MW)"])

    return np.allclose(gen_sum, wind + solar), np.sum(gen_sum), np.sum(wind + solar)


def check_storage_status(instance):
    both_zero = []  # to record units where both charge & discharge = 0

    for storage_name, discharge_values in instance["Storage discharging rates (MW)"].items():
        discharge = np.array(discharge_values)
        charge = np.array(instance["Storage charging rates (MW)"][storage_name])

        # Condition 1: both nonzero at same time (NOT ALLOWED)
        both_nonzero = (charge > 0) & (discharge > 0)
        if np.any(both_nonzero):
            return False, storage_name, "charging & discharging simultaneously"

        # Condition 2: charging when discharge is non-zero (NOT ALLOWED)
        charge_when_discharge = (charge > 0) & (discharge != 0)
        if np.any(charge_when_discharge):
            return False, storage_name, "charging while discharging is non-zero"

        # Condition 3: discharging when charge is non-zero (NOT ALLOWED)
        discharge_when_charge = (discharge > 0) & (charge != 0)
        if np.any(discharge_when_charge):
            return False, storage_name, "discharging while charge is non-zero"

        # Condition 4: both zero → record for later
        both_zero_mask = (charge == 0) & (discharge == 0)
        if np.any(both_zero_mask):
            both_zero.append(storage_name)

    return True, both_zero
