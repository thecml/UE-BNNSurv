__author__='Christian Marius Lillelund'
__author_email__='chr1000@gmail.com'

import tensorflow as tf
from typing import List
import numpy as np
from .utility import InputFunction, CoxPHLoss, make_mlp_model, make_vi_model, make_mcd_model

class MLP():
    """
    Multilayer perceptron with non-stochastic weights.
    """
    def __init__(self, layers:List=[], dropout_rate:float=0.25,
                 regularization_pen:float=0, batch_size:int=32,
                 num_epochs:int=10) -> None:

        """
        Initialize MLP class.
        
        :param layers: list of hidden layers and number of units
        :param dropout_rate: applied dropout rate
        :param regularization_pen: l2 regularization penalty term
        :param batch_size: batch size to use
        :param event_times: time horizon for prediction
        :param num_epochs: number of training epochs
        """
        super().__init__()
        self.layers = layers
        self.dropout = dropout_rate
        self.l2_reg = regularization_pen
        self.activation_fn = "relu"
        self.batch_size = batch_size
        self.n_epochs= num_epochs
        self.loss_fn = CoxPHLoss()
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = None
    
    def fit(self, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray) -> None:
        """
        Fit the MLP survival regressor.
        
        :param X: list of featutes in shape [n_samples, n_features]
        :param y_time: time to event or censoring
        :param y_event: boolean indicating whether the event time is censored or the event occured
        """
        train_ds = InputFunction(X, y_time, y_event, batch_size=self.batch_size,
                                 drop_last=True, shuffle=True)()
        self.model = make_mlp_model(input_shape=X.shape[1:], output_dim=1,
                                    layers=self.layers, activation_fn=self.activation_fn,
                                    dropout_rate=self.dropout, regularization_pen=self.l2_reg)
        for _ in range(self.n_epochs):
            for x, y in train_ds:
                y_event = tf.expand_dims(y["label_event"], axis=1)
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                with tf.name_scope("gradients"):
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def predict_risk(self, X) -> np.ndarray:
        """
        Predict risk of event.
        
        :param X: list of features with shape [n_samples, n_features]
        :return: predicted risk with shape [n_samples]
        """
        preds = self.model(X, training=False)
        return preds.numpy().flatten()

class VI():
    """
    Multilayer perceptron with stochastic weights using variational approximation.
    """
    def __init__(self, layers:List=[], dropout_rate:float=0.25,
                 regularization_pen:float=0, batch_size:int=32,
                 num_epochs:int=10) -> None:
        """
        Initialize VI class.
        
        :param layers: list of hidden layers and number of units
        :param dropout_rate: applied dropout rate
        :param regularization_pen: l2 regularization penalty term
        :param batch_size: batch size to use
        :param num_epochs: number of training epochs
        """
        super().__init__()
        self.layers = layers
        self.dropout = dropout_rate
        self.l2_reg = regularization_pen
        self.activation_fn = "relu"
        self.batch_size = batch_size
        self.n_epochs= num_epochs
        self.loss_fn = CoxPHLoss()
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = None

    def fit(self, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray) -> None:
        """
        Fit the VI survival regressor.
        
        :param X: list of featutes in shape [n_samples, n_features]
        :param y_time: time to event or censoring
        :param y_event: boolean indicating whether the event time is censored or the event occured
        """
        train_ds = InputFunction(X, y_time, y_event, batch_size=self.batch_size,
                                 drop_last=True, shuffle=True)()
        self.model = make_vi_model(n_train_samples=X.shape[0], input_shape=X.shape[1:],
                                   output_dim=2, layers=self.layers, activation_fn=self.activation_fn,
                                   dropout_rate=self.dropout, regularization_pen=self.l2_reg)
        for _ in range(self.n_epochs):
            for x, y in train_ds:
                y_event = tf.expand_dims(y["label_event"], axis=1)
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    cox_loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                    loss = cox_loss + tf.reduce_mean(self.model.losses) # CoxPHLoss + KL-divergence
                with tf.name_scope("gradients"):
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def predict_risk(self, X, post_samples:int=100) -> np.ndarray:
        """
        Predict risk of event.
        
        :param X: list of features with shape [n_samples, n_features]
        :param post_samples: number of samples to draw from posterior
        :return: predicted risk with shape [n_samples]
        """
        logits_cpd = np.zeros((post_samples, len(X)), dtype=np.float32)
        for i in range(0, post_samples):
            logits_cpd[i,:] = np.reshape(self.model(X, training=False).sample(), len(X))
        preds = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
        return preds.numpy().flatten()

class MCD():
    """
    Multilayer perceptron with stochastic weights using Monte Carlo Dropout.
    """
    def __init__(self, layers:List=[], dropout_rate:float=0.25,
                 regularization_pen:float=0, batch_size:int=32,
                 num_epochs:int=10) -> None:
        super().__init__()
        """
        Initialize MCD class.
        
        :param layers: list of hidden layers and number of units
        :param dropout_rate: applied dropout rate
        :param regularization_pen: l2 regularization penalty term
        :param batch_size: batch size to use
        :param num_epochs: number of training epochs
        """    
        self.layers = layers
        self.dropout = dropout_rate
        self.l2_reg = regularization_pen
        self.activation_fn = "relu"
        self.batch_size = batch_size
        self.n_epochs= num_epochs
        self.loss_fn = CoxPHLoss()
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = None

    def fit(self, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray) -> None:
        """
        Fit the MCD survival regressor.
        
        :param X: list of featutes in shape [n_samples, n_features]
        :param y_time: time to event or censoring
        :param y_event: boolean indicating whether the event time is censored or the event occured
        """
        train_ds = InputFunction(X, y_time, y_event, batch_size=self.batch_size,
                                 drop_last=True, shuffle=True)()
        self.model = make_mcd_model(input_shape=X.shape[1:], output_dim=2,
                                    layers=self.layers, activation_fn=self.activation_fn,
                                    dropout_rate=self.dropout, regularization_pen=self.l2_reg)
        for _ in range(self.n_epochs):
            for x, y in train_ds:
                y_event = tf.expand_dims(y["label_event"], axis=1)
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                with tf.name_scope("gradients"):
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def predict_risk(self, X, post_samples:int=100) -> np.ndarray:
        """
        Predict risk of event.
        
        :param X: list of features with shape [n_samples, n_features]
        :param post_samples: number of samples to draw from posterior
        :return: predicted risk with shape [n_samples]
        """
        logits_cpd = np.zeros((post_samples, len(X)), dtype=np.float32)
        for i in range(0, post_samples):
            logits_cpd[i,:] = np.reshape(self.model(X, training=False).sample(), len(X))
        preds = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
        return preds.numpy().flatten()
