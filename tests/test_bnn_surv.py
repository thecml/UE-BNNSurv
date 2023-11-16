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

def test_bnn_surv():
    # Load data
    X, y = load_whas500()
    y = convert_to_structured(y['lenfol'], y['fstat'])
    
    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)

    # Make model
    model = models.MCD(layers=[32, 32])
    
    # Train model
    model.fit(X_train, t_train, e_train)
    
    # Make predictions
    preds = model.predict_risk(X_test)
    
    assert len(preds) != 0
