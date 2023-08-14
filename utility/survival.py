import numpy as np
from sksurv.linear_model.coxph import BreslowEstimator

def compute_survival_scale(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState()

    # generate hazard scale
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    return scale

def compute_survival_times(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState(0)

    # generate survival time
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    u = rnd.uniform(low=0, high=1, size=risk_scores.shape[0])
    t = -np.log(u) / scale

    return t

def compute_survival_times_with_censoring(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState(0)

    # generate survival time
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    u = rnd.uniform(low=0, high=1, size=risk_scores.shape[0])
    t = -np.log(u) / scale

    # generate time of censoring
    prob_censored = 1 - e_train.sum()/len(e_train)
    qt = np.quantile(t, 1.0 - prob_censored)
    c = rnd.uniform(low=t.min(), high=qt)

    # apply censoring
    observed_event = t <= c
    observed_time = np.where(observed_event, t, c)
    return observed_time, observed_event

def convert_to_structured(T, E):
    # dtypes for conversion
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}

    # concat of events and times
    concat = list(zip(E, T))

    # return structured array
    return np.array(concat, dtype=default_dtypes)

def get_breslow_survival_times(model, X_train, X_test, e_train, t_train, runs):
    train_predictions = model.predict(X_train, verbose=0).reshape(-1)
    breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
    model_cpd = np.zeros((runs, len(X_test)))
    for i in range(0, runs):
        model_cpd[i,:] = np.reshape(model.predict(X_test, verbose=0), len(X_test))
    event_times = breslow.get_survival_function(model_cpd[0])[0].x
    breslow_surv_times = np.zeros((len(X_test), runs, len(event_times)))
    for i in range(0, runs):
        surv_fns = breslow.get_survival_function(model_cpd[i,:])
        for j, surv_fn in enumerate(surv_fns):
            breslow_surv_times[j,i,:] = surv_fn.y
    return breslow_surv_times