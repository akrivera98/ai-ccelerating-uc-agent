# dir used
input_dir = '/app/input_data/'
output_dir = '/app/output/'
submission_dir = '/app/ingested_program/'

# Libraries for ingestion step
import pandas as pd
import dill
import time
import json
import os 

def order_key(s):
    numb = s.split('_')
    ident = numb[1] + numb[2][-1] + numb[3]
    return int(ident)

def get_model():
    # Upload participant's model
    with open(submission_dir+'model.dill', 'rb') as file:
        model = dill.load(file) 
    return model

def get_test_data():
    x_fea = {}
 
    list_ins = sorted(os.listdir(input_dir),key=order_key)
    for index, instance in enumerate(list_ins):
        x_fea[index] = pd.read_excel(input_dir+instance+"/explanatory_variables.xlsx",sheet_name=None,index_col=0)
    return x_fea

def main():
    print('Reading Reference Data')
    features = get_test_data()
    print('-' * 10)
    print('Running Prediction')
    model = get_model()
    start = time.time()
    # Prediction IN
    try:
        status = model.predict(features)
    except Exception as e:
        print("HUH")
        print("Something went wrong: ",e)
        raise e
    duration = time.time() - start
    print('-' * 10)
    print(f'Completed Prediction. Total duration: {duration}')
    with open(output_dir+'prediction.dill', 'wb') as file:
        dill.dump(status,file)
    with open(output_dir+'metadata.json', 'w+') as f:
        json.dump({'duration': duration}, f)
    print()
    print('Ingestion Program finished. Moving on to scoring')

if __name__ == '__main__':
    main()

    

    

    