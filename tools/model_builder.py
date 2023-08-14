import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utility.loss import CoxPHLoss
from utility.metrics import CindexMetric
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from auton_survival.estimators import SurvivalModel
from auton_survival import DeepCoxPH

class MonteCarloDropout(tf.keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

tfd = tfp.distributions
tfb = tfp.bijectors

def normal_loc(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def normal_loc_scale(params):
    return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))

def normal_fs(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    return CoxPHSurvivalAnalysis(alpha=0.0001, n_iter=n_iter, tol=tol)

def make_rsf_model(config):
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    min_samples_split = config['min_samples_split']
    min_samples_leaf =  config['min_samples_leaf']
    max_features = config['max_features']
    return RandomSurvivalForest(random_state=0,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)

def make_coxnet_model(config):
    l1_ratio = config['l1_ratio']
    alpha_min_ratio = config['alpha_min_ratio']
    n_alphas = config['n_alphas']
    normalize = config['normalize']
    tol = config['tol']
    max_iter = config['max_iter']
    return CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                  l1_ratio=l1_ratio,
                                  alpha_min_ratio=alpha_min_ratio,
                                  n_alphas=n_alphas,
                                  normalize=normalize,
                                  tol=tol,
                                  max_iter=max_iter)
def make_dsm_model(config):
    layers = config['network_layers']
    n_iter = config['n_iter']
    return SurvivalModel('dsm', random_seed=0, iters=n_iter,
                         layers=layers, distribution='Weibull', max_features='sqrt')

def make_dcph_model(config):
    layers = config['network_layers']
    return DeepCoxPH(layers=layers)
    
def make_mlp_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
                
    output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=output)
                
    params = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
    if output_dim == 2: # If 2, then model aleatoric uncertain.
        dist = tfp.layers.DistributionLambda(normal_loc_scale)(params)
        model = tf.keras.Model(inputs=inputs, outputs=dist)
    else: # Do not model aleatoric uncertain
        output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_vi_model(n_train_samples, input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                 activity_regularizer=tf.keras.regularizers.L2(regularization_pen),
                                                 kernel_divergence_fn=kernel_divergence_fn,
                                                 bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(inputs)
            else:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                activity_regularizer=tf.keras.regularizers.L2(regularization_pen),
                                kernel_divergence_fn=kernel_divergence_fn,
                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(hidden)
            else:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    params = tfp.layers.DenseFlipout(output_dim,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn)(hidden)
    if output_dim == 2: # If 2, then model both aleatoric and epistemic uncertain.
        dist = tfp.layers.DistributionLambda(normal_loc_scale)(params)
    else: # model only epistemic uncertain.
        dist = tfp.layers.DistributionLambda(normal_loc)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    return model

def make_mcd_model(input_shape, output_dim, layers,
                   activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
    params = tf.keras.layers.Dense(output_dim)(hidden)
    dist = tfp.layers.DistributionLambda(normal_loc_scale)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    return model