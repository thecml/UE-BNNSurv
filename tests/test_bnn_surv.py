from bnnsurv import models
import pandas as pd
import numpy as np
from sksurv.datasets import load_whas500
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_time_event_split(y):
    y_t = np.array(y['time'])
    y_e = np.array(y['event'])
    return (y_t, y_e)

def convert_to_structured(T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def test_mlp():
    X, y = load_whas500()
    y = convert_to_structured(y['lenfol'], y['fstat'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    t_train, e_train = make_time_event_split(y_train)
    
    lower, upper = np.percentile(t_train[t_train.dtype.names], [10, 90])
    event_times = np.arange(lower, upper+1)
    
    model = models.MLP(layers=[32, 32], num_epochs=1)
    model.fit(X_train, t_train, e_train)
    preds_risk = model.predict_risk(X_test)
    preds_surv = model.predict_survival(X_test, event_times)

    assert len(preds_risk) != 0
    assert len(preds_surv) != 0

def test_vi():
    X, y = load_whas500()
    y = convert_to_structured(y['lenfol'], y['fstat'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    t_train, e_train = make_time_event_split(y_train)
    
    lower, upper = np.percentile(t_train[t_train.dtype.names], [10, 90])
    event_times = np.arange(lower, upper+1)
    
    model = models.VI(layers=[32, 32], num_epochs=1)
    model.fit(X_train, t_train, e_train)
    preds_risk = model.predict_risk(X_test)
    preds_surv = model.predict_survival(X_test, event_times)

    assert len(preds_risk) != 0
    assert len(preds_surv) != 0

def test_mcd():
    X, y = load_whas500()
    y = convert_to_structured(y['lenfol'], y['fstat'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    t_train, e_train = make_time_event_split(y_train)
    
    lower, upper = np.percentile(t_train[t_train.dtype.names], [10, 90])
    event_times = np.arange(lower, upper+1)
    
    model = models.MCD(layers=[32, 32], num_epochs=1)
    model.fit(X_train, t_train, e_train)
    preds_risk = model.predict_risk(X_test)
    preds_surv = model.predict_survival(X_test, event_times)

    assert len(preds_risk) != 0
    assert len(preds_surv) != 0

