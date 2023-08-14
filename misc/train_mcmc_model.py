import pandas as pd
import numpy as np
import tensorflow as tf
from tools.data_loader import load_veterans_ds, prepare_veterans_ds, load_cancer_ds, \
                        prepare_cancer_ds, load_aids_ds, prepare_aids_ds
from sklearn.preprocessing import StandardScaler
from utility import InputFunction, CindexMetric, CoxPHLoss, _make_riskset, sample_hmc, convert_to_structured
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tensorflow_probability as tfp
from utility import _TFColor
import seaborn as sns
import pickle
from sksurv.metrics import concordance_index_censored, integrated_brier_score

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = tf.float32
n_chains = 5

if __name__ == "__main__":
    # Load data
    X_train, _, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, _, t_test, e_train, _, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Split training data in observed/unobserved
    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=DTYPE)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=DTYPE)
    x_obs = tf.convert_to_tensor(X_train[e_train], dtype=DTYPE)
    x_cens = tf.convert_to_tensor(X_train[~e_train], dtype=DTYPE)

    n_dims = x_cens.shape[1]

    obs_model = tfd.JointDistributionSequentialAutoBatched([
            tfd.Normal(loc=tf.zeros([1]), scale=tf.ones([1])), # alpha
            tfd.Normal(loc=tf.zeros([n_dims,1]), scale=tf.ones([n_dims,1])), # beta
            lambda beta, alpha: tfd.Exponential(rate=1/tf.math.exp(tf.transpose(x_obs)*beta + alpha))]
        )

    def log_prob(x_obs, x_cens, y_obs, y_cens, alpha, beta):
        lp = obs_model.log_prob([alpha, beta, y_obs])
        potential = exponential_lccdf(x_cens, y_cens, alpha, beta)
        return lp + potential

    def exponential_lccdf(x_cens, y_cens, alpha, beta):
        return tf.reduce_sum(-y_cens / tf.exp(tf.transpose(x_cens)*beta + alpha))

    n_chains = 5
    number_of_steps = 10000
    number_burnin_steps = int(number_of_steps/10)

    # Sample from the prior
    initial_coeffs = obs_model.sample(1)

    # Run sampling for number of chains
    unnormalized_post_log_prob = lambda *args: log_prob(x_obs, x_cens, y_obs, y_cens, *args)
    chains = [sample_hmc(unnormalized_post_log_prob, [tf.zeros_like(initial_coeffs[0]),
                                                      tf.zeros_like(initial_coeffs[1])],
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps) for _ in range(n_chains)]

    # Calculate target accept prob
    for chain_id in range(n_chains):
        log_accept_ratio = chains[chain_id][1][1][number_burnin_steps:]
        target_accept_prob = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.))).numpy()
        print(f'Target acceptance probability for {chain_id}: {round(100*target_accept_prob)}%')

    # Calculate accepted rate
    plt.figure(figsize=(10,6))
    for chain_id in range(n_chains):
        accepted_samples = chains[chain_id][1][0][number_burnin_steps:]
        print(f'Acceptance rate chain for {chain_id}: {round(100*np.mean(accepted_samples), 2)}%')
        n_accepted_samples = len(accepted_samples)
        n_bins = int(n_accepted_samples/100)
        sample_indicies = np.linspace(0, n_accepted_samples, n_bins)
        means = [np.mean(accepted_samples[:int(idx)]) for idx in sample_indicies[5:]]
        plt.plot(np.arange(len(means)), means)
    plt.show()

    # Take mean of combined chains to get alpha and beta values
    chains = [chain[0] for chain in chains] # leave out traces

    chain_index = 0
    samples_index = 0
    beta_index = 1
    n_dims = chains[chain_index][beta_index].shape[2] # get n dims from first chain
    chains_t = list(map(list, zip(*chains)))
    chains_samples = [tf.squeeze(tf.concat(samples, axis=0)) for samples in chains_t]
    alpha = tf.reduce_mean(chains_samples[0]).numpy().flatten()
    betas = tf.reduce_mean(chains_samples[1], axis=0).numpy().flatten()

    # Make predictions on test set
    predict_func = lambda data: np.exp(alpha + np.dot(betas, np.transpose(data)))
    test_preds = np.zeros((len(X_test)))
    for i, data in enumerate(X_test):
        test_preds[i] = predict_func(data)

    # Calculate concordance index
    c_index = concordance_index_censored(e_test, t_test, -test_preds)[0]

    # Calculate Brier score
    lower, upper = np.percentile(t_test[t_test.dtype.names], [10, 90])
    times = np.arange(lower, upper+1)
    estimate = np.zeros((len(X_test), len(times)))
    for i, data_obs in enumerate(X_test):
        pred_lambda = predict_func(data)
        surv_prob = np.exp(-times/pred_lambda) # survial function
        estimate[i] = surv_prob
    y_train_struc = convert_to_structured(t_train, e_train)
    y_test_struc = convert_to_structured(t_test, e_test)
    ibs = integrated_brier_score(y_train_struc, y_test_struc, estimate, times)
    print(f"Training completed, test C-index/BS: {round(c_index, 4)}/{round(ibs, 4)}")

    # Save chains
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    with open(f'{root_dir}/models/mcmc_chains.pkl', 'wb') as fp:
        pickle.dump(chains, fp)