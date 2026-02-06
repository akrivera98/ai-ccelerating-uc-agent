# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "dill",
#     "pandas",
#     "juliacall",
# ]
# ///

import gzip
import json
import dill
import pandas as pd
import time 
import sys 
import os

reference_dir = '/app/input/ref/'
prediction_dir = '/app/input/res/'
score_dir = '/app/output/'

print('Loading Julia')
from juliacall import Main
Main.include("/app/scripts/ED_Script.jl")
print('-' * 10)

def order_key(s):
    numb = s.split('_')
    ident = numb[1] + numb[2][-1] + numb[3]
    return int(ident)

def get_prediction():
    # Upload participant's model
    with open(prediction_dir+'prediction.dill', 'rb') as file:
        prediction = dill.load(file) 
    return prediction

def get_input_data():
    data_in = {}
    data_index = {}
    list_ins = sorted(os.listdir(reference_dir),key=order_key)

    for index, instance in enumerate(list_ins):
        with gzip.open(reference_dir+instance+"/InputData.json.gz","r") as f:
            in_json = f.read().decode("utf-8")
            data_in[instance] = json.loads(in_json)
            data_index[instance] = index
    return data_in, data_index

def main():
    print('Scoring Program')
    print('-' * 10)
    print('Reading Input Data')
    data_in, data_index = get_input_data()
    print('-' * 10)
    print('Reading Reference Data')
    prediction = get_prediction()
    print('-' * 10)
    meta = pd.read_csv("/app/scripts/instances_actual_data.csv",index_col=0)
    score = pd.DataFrame()
    # Duration training
    with open(prediction_dir+'metadata.json') as f:
        duration_train = json.load(f).get('duration', -1)
    print('Evaluating solution by means of ED')
    start = time.time()
    for case in data_in.keys():
        for g in data_in[case]["Generators"].keys():
            if data_in[case]["Generators"][g]["Type"] == "Thermal":
                data_in[case]["Generators"][g]["Commitment status"] = [int(round(item)) for item in prediction[data_index[case]][g].tolist()]

        # Save the data to be read by python
        json_str = json.dumps(data_in[case])                                # 1. string (i.e. JSON)
        json_bytes = json_str.encode('utf-8')                               # 2. bytes (i.e. UTF-8)
        with gzip.open("InputData.json.gz", 'w') as fout: # 3. fewer bytes (i.e. gzip)
            fout.write(json_bytes)  
        
        print(f"Solving {case}")
        # Solving the Economic Dispatch
        run_time,obj_value,status,solution = Main.run_ED("InputData.json.gz")
        print(f'Solved {case}, status: {status}')

        real_value = meta[meta.index==case]["objetive function ($)"].tolist()[0]
        real_time  = meta[meta.index==case]["run time (s)"].tolist()[0]
        try:# Storing relevant data
            grow_rate = int((obj_value-real_value)/real_value*100/0.1) if int((obj_value-real_value)/real_value*100/0.1) >= 0 else 0
        except:
            grow_rate = 0
        score.loc[case,"hybrid_solution"] = round(obj_value/1e6,2)
        score.loc[case,"real_solution"] = round(real_value/1e6,2)
        score.loc[case,"points"] = 100 - grow_rate if 100 - grow_rate > 0 else 0
        score.loc[case,"real_time"] = real_time
        score.loc[case,"ED_time"] = run_time
        score.loc[case,"final_time"] = round(run_time + duration_train/len(data_in.keys()),2)
        score.loc[case,"bonus"] = 10 if (score.at[case,"final_time"] < 0.01*real_time) and (real_time>= 100) and (score.loc[case,"points"]>0) else 0
        
        # Penalties due to time
        if (score.at[case,"final_time"] >= 0.1*real_time) and (score.at[case,"final_time"] < 0.5*real_time) and (real_time >= 100):
            score.loc[case,"penalty"] = -score.loc[case,"points"]/2 
        elif (score.at[case,"final_time"] >= 0.5*real_time) and (real_time >= 100):
            score.loc[case,"penalty"] = -score.loc[case,"points"]
        else:
            score.loc[case,"penalty"] = 0

    score.to_csv(score_dir+"scoring_results.csv")
    # Final steps
    duration = time.time() - start
    print('-' * 10)
    print(f'No more evaluations. Total duration: {duration:.2f}s')
    print('Checking Accuracy') 
    scoring = score["points"].sum() + score["bonus"].sum() + score["penalty"].sum()
    duration_test = score["final_time"].sum()
    print(f'Average score: {scoring/len(data_in.keys())}')
    scores = {
    'accuracy': scoring,
    'duration': duration_test+duration_train}
    with open(score_dir+'scores.json', 'w') as score_file:
        score_file.write(json.dumps(scores))
    print('-' * 10)

if __name__ == "__main__":
    main()