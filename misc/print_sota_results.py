import pandas as pd
import paths as pt
from pathlib import Path

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(3)
    
    model_names = results['ModelName'].unique()
    dataset_names = ["WHAS500", "SEER", "GBSG2", "FLCHAIN", "SUPPORT", "METABRIC"]
    model_citations = ['\cite{cox_regression_1972}', '\cite{simon_regularization_2011}',
                       '\cite{ishwaran_random_2008}', '\cite{nagpal_deep_2021}',
                       '\cite{katzman_deepsurv_2018}']

    print(dataset_names)
    print(model_names)
    
    for dataset_name in dataset_names:
        for index, (model_name, model_citation) in enumerate(zip(model_names, model_citations)):
            text = ""
            t_train = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TrainTime'])
            ci = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestCI'])
            ctd = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestCTD'])
            ibs = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestIBS'])
            loss = float(results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]['TestLoss'])
            if loss != loss:
                loss = "NA"
            if model_name == "Cox":
                model_name = "CoxPH"
            if model_name == "DCPH":
                model_name = "DeepSurv"
            text += f"{model_name} {model_citation} & "
            text += f"{t_train} & {ci} & {ctd} & {ibs} & {loss} \\\\"
            print(text)
        print()
        
        