from tools.data_loader import (BaseDataLoader, FlchainDataLoader, GbsgDataLoader, MetabricDataLoader,
                               SupportDataLoader, WhasDataLoader, AidsDataLoader, SeerDataLoader)
from tools.preprocessor import Preprocessor
from typing import Tuple
import numpy as np

def get_data_loader(dataset_name:str) -> BaseDataLoader:
    if dataset_name == "FLCHAIN":
        return FlchainDataLoader()
    elif dataset_name == "SEER":
        return SeerDataLoader()
    elif dataset_name == "GBSG2":
        return GbsgDataLoader()
    elif dataset_name == "METABRIC":
        return MetabricDataLoader()
    elif dataset_name == "SUPPORT":
        return SupportDataLoader()
    elif dataset_name == "WHAS500":
        return WhasDataLoader()
    elif dataset_name == "AIDS":
        return AidsDataLoader()
    else:
        raise ValueError("Data loader not found")

def scale_data(X_train, X_test, cat_features, num_features) -> Tuple[np.ndarray, np.ndarray]:
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    return (X_train, X_test)

def make_time_event_split(y):
    y_t = np.array(y['time'])
    y_e = np.array(y['event'])
    return (y_t, y_e)