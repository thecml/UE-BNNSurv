import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Sequence, Tuple, Dict, Iterable
import numpy as np

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

class CoxPHLoss(tf.keras.losses.Loss):
    """Negative partial log-likelihood of Cox's proportional hazards model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,
             y_true: Sequence[tf.Tensor],
             y_pred: tf.Tensor) -> tf.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_true : list|tuple of tf.Tensor
            The first element holds a binary vector where 1
            indicates an event 0 censoring.
            The second element holds the riskset, a
            boolean matrix where the `i`-th row denotes the
            risk set of the `i`-th instance, i.e. the indices `j`
            for which the observer time `y_j >= y_i`.
            Both must be rank 2 tensors.
        y_pred : tf.Tensor
            The predicted outputs. Must be a rank 2 tensor.

        Returns
        -------
        loss : tf.Tensor
            Loss for each instance in the batch.
        """
        event, riskset = y_true
        predictions = y_pred
        
        pred_shape = predictions.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)

        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal ranvk of event (received %s)" % (
                pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)

        event = tf.cast(event, predictions.dtype)
        predictions = safe_normalize(predictions)

        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        pred_t = tf.transpose(predictions)

        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()

        losses = tf.math.multiply(event, rr - predictions)

        return losses

def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm

def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)

        # for numerical stability, substract the maximum value
        # before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output

class InputFunction:
    """Callable input function that computes the risk set for each batch.

    Parameters
    ----------
    data : np.ndarray, shape=(n_samples, n_fts)
        Obs data.
    time : np.ndarray, shape=(n_samples,)
        Observed time.
    event : np.ndarray, shape=(n_samples,)
        Event indicator.
    batch_size : int, optional, default=32
        Number of samples per batch.
    drop_last : int, optional, default=False
        Whether to drop the last incomplete batch.
    shuffle : bool, optional, default=False
        Whether to shuffle data.
    seed : int, optional, default=89
        Random number seed.
    """

    def __init__(self,
                 data: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray,
                 batch_size: int = 32,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 seed: int = 0) -> None:
        self.data = data
        self.time = time
        self.event = event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def size(self) -> int:
        """Total number of samples."""
        return self.data.shape[0]

    def steps_per_epoch(self) -> int:
        """Number of batches for one epoch."""
        return int(np.floor(self.size() / self.batch_size))

    def _get_data_batch(self, index: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute risk set for samples in batch."""
        time = self.time[index]
        event = self.event[index]
        data = self.data[index]

        labels = {
            "label_event": event.astype(np.int32),
            "label_time": time.astype(np.float32),
            "label_riskset": _make_riskset(time)
        }
        return data, labels

    def _iter_data(self) -> Iterable[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Generator that yields one batch at a time."""
        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)

        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch_size)]
            yield self._get_data_batch(idx)

        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            yield self._get_data_batch(idx)

    def _get_shapes(self) -> Tuple[tf.TensorShape, Dict[str, tf.TensorShape]]:
        """Return shapes of data returned by `self._iter_data`."""
        batch_size = self.batch_size if self.drop_last else None
        _, n_features = self.data.shape
        rows = tf.TensorShape([batch_size, n_features])

        labels = {k: tf.TensorShape((batch_size,))
                  for k in ("label_event", "label_time")}
        labels["label_riskset"] = tf.TensorShape((batch_size, batch_size))
        return rows, labels

    def _get_dtypes(self) -> Tuple[tf.DType, Dict[str, tf.DType]]:
        """Return dtypes of data returned by `self._iter_data`."""
        labels = {"label_event": tf.int32,
                  "label_time": tf.float32,
                  "label_riskset": tf.bool}
        return tf.float32, labels

    def _make_dataset(self) -> tf.data.Dataset:
        """Create dataset from generator."""
        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            self._get_dtypes(),
            self._get_shapes()
        )
        return ds

    def __call__(self) -> tf.data.Dataset:
        return self._make_dataset()

def _make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set