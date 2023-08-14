import pandas as pd
import paths as pt
from pathlib import Path
import glob
import os

if __name__ == "__main__":
    path = pt.RESULTS_DIR
    all_files = glob.glob(os.path.join(path , "baysurv*.csv"))
    
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    results = pd.concat(li, axis=0, ignore_index=True)
            
    results = results.round(3)
    last_obs = results.groupby(['ModelName', 'DatasetName'])['TrainTime', 'TestCI', 'TestCTD', 'TestIBS', 'TestLoss'].last()
    train_times = results.groupby(['ModelName', 'DatasetName'])['TrainTime'].sum()
    results = pd.concat([last_obs.drop('TrainTime', axis=1), train_times], axis=1).reset_index()
    
    model_names = ['MLP', 'VI', 'MCD']
    dataset_name = ["WHAS500", "SEER", "GBSG2", "FLCHAIN", "SUPPORT", "METABRIC"]
    
    for dataset_name in dataset_name:
        for index, model_name in enumerate(model_names):
            if index > 0:
                text = "+ "
            else:
                text = ""
            t_train = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TrainTime'])
            ci = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestCI'])
            ctd = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestCTD'])
            ibs = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestIBS'])
            loss = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestLoss'])
            if model_name == "MLP":
                model_name = "Baseline (MLP)"
            text += f"{model_name} & "
            text += f"{t_train} & {ci} & {ctd} & {ibs} & {loss} \\\\"
            print(text)
        print()
        