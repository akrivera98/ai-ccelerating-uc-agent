import os
import json
import gzip
import re
import pandas as pd
import numpy as np

print(os.getcwd())
reference_dir = "data/starting_kit/Train_Data"
NUM_INSTANCES = 3


def order_key(s):
    # numb = s.split('_')
    m = re.search(r"_(\d+)$", s)

    # return int(numb[1])
    return int(m.group(1))


def get_input_data():
    data_in = {}
    all_files = sorted(os.listdir(reference_dir))
    list_ins = [file for file in all_files if not file.startswith(".")]
    list_ins = sorted(list_ins, key=order_key)[:NUM_INSTANCES]
    data_index = []
    for instance in list_ins:
        # with open(reference_dir + instance + "/metadata_i.json", "r") as f:
        #     data_index[index] = json.load(f)["i_done"]
        with gzip.open(reference_dir + f"/{instance}/InputData.json.gz", "r") as f:
            in_json = f.read().decode("utf-8")
            data_in[instance] = json.loads(in_json)
    return data_in, data_index


def create_dummy_status(data_index, generator_names):
    status = {}
    n_periods = 72

    for idx in data_index:
        dummy_array = np.random.rand(
            n_periods, len(generator_names)
        )  # shape like predictions
        status[idx] = pd.DataFrame(
            dummy_array, index=range(n_periods), columns=generator_names
        )

    return status


data_in, data_index = get_input_data()
status = create_dummy_status(data_in.keys(), data_in[1]["Generators"].keys())

for case in data_in.keys():
    for g in data_in[case]["Generators"].keys():
        if data_in[case]["Generators"][g]["Type"] == "Thermal":
            data_in[case]["Generators"][g]["Commitment status"] = [
                int(round(item)) for item in status[case][g].tolist()
            ]
