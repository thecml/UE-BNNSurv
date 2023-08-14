import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids, load_whas500, load_flchain
from sklearn.model_selection import train_test_split
import shap
from abc import ABC, abstractmethod
from typing import Tuple, List
from tools.preprocessor import Preprocessor
import paths as pt
from pathlib import Path
from utility.survival import convert_to_structured

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y: np.ndarray = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None

    @abstractmethod
    def load_data(self) -> None:
        """Loads the data from a data set at startup"""

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train['Time'])
        t_valid = np.array(y_valid['Time'])
        t_test = np.array(y_test['Time'])
        e_train = np.array(y_train['Event'])
        e_valid = np.array(y_valid['Event'])
        e_test = np.array(y_test['Event'])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

    def get_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        This method returns the features and targets
        :return: X and y
        """
        return self.X, self.y

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['category']).columns.tolist()

    def prepare_data(self, train_size: float = 0.7) -> Tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """
        This method prepares and splits the data from a data set
        :param train_size: the size of the train set
        :return: a split train and test dataset
        """
        X = self.X
        y = self.y
        cat_features = self.cat_features
        num_features = self.num_features

        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=0)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=0)

        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
        transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                       one_hot=True, fill_value=-1)
        X_train = transformer.transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

        X_train = np.array(X_train, dtype=np.float32)
        X_valid = np.array(X_valid, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

class SeerDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset
    """
    def load_data(self):
        path = Path.joinpath(pt.DATA_DIR, 'seer.csv')
        data = pd.read_csv(path)

        data = data.loc[data['Survival Months'] > 0]

        outcomes = data.copy()
        outcomes['event'] =  data['Status']
        outcomes['time'] = data['Survival Months']
        outcomes = outcomes[['event', 'time']]
        outcomes.loc[outcomes['event'] == 'Alive', ['event']] = 0
        outcomes.loc[outcomes['event'] == 'Dead', ['event']] = 1

        data = data.drop(['Status', "Survival Months"], axis=1)

        obj_cols = data.select_dtypes(['bool']).columns.tolist() \
                + data.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            data[col] = data[col].astype('category')

        self.X = pd.DataFrame(data)

        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

class SupportDataLoader(BaseDataLoader):
    """
    Data loader for SUPPORT dataset
    """
    def load_data(self):
        path = Path.joinpath(pt.DATA_DIR, 'support.feather')
        data = pd.read_feather(path)

        data = data.loc[data['duration'] > 0]

        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]

        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                      'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

        self.num_features = num_feats
        self.cat_features = []
        self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

class NhanesDataLoader(BaseDataLoader):
    """
    Data loader for NHANES dataset
    """
    def load_data(self):
        X, y = shap.datasets.nhanesi()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        event = np.array([True if x > 0 else False for x in y])
        time = np.array(abs(y))
        self.y = convert_to_structured(time, event)

        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class AidsDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_aids()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')
        self.X = pd.DataFrame(X)

        self.y = convert_to_structured(y['time'], y['censor'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class GbsgDataLoader(BaseDataLoader):
    def load_data(self) -> BaseDataLoader:
        X, y = load_gbsg2()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['time'], y['cens'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class WhasDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_whas500()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['lenfol'], y['fstat'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class FlchainDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_flchain()
        X['event'] = y['death']
        X['time'] = y['futime']

        X = X.loc[X['time'] > 0]
        self.y = convert_to_structured(X['time'], X['event'])
        X = X.drop(['event', 'time'], axis=1).reset_index(drop=True)

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class MetabricDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        path = Path.joinpath(pt.DATA_DIR, 'metabric.feather')
        data = pd.read_feather(path)

        data = data.loc[data['duration'] > 0]

        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]

        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x8'] \
                     + ['x4', 'x5', 'x6', 'x7']

        self.num_features = num_feats
        self.cat_features = []
        self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self