import tensorflow as tf
import numpy as np
from utility.metrics import CindexMetric, CindexTdMetric, IbsMetric
from utility.survival import convert_to_structured
from time import time

class Trainer:
    def __init__(self, model, model_type, train_dataset, valid_dataset,
                 test_dataset, optimizer, loss_function, num_epochs,
                 event_times):
        self.num_epochs = num_epochs
        self.model = model
        self.model_type = model_type

        self.train_ds = train_dataset
        self.valid_ds = valid_dataset
        self.test_ds = test_dataset

        self.optimizer = optimizer
        self.loss_fn = loss_function

        self.train_loss_scores = list()
        self.valid_loss_scores, self.valid_ci_scores = list(), list()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_cindex_metric = CindexMetric()
        self.train_ctd_metric = CindexTdMetric()
        self.train_ibs_metric = IbsMetric(event_times)
        self.valid_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.valid_cindex_metric = CindexMetric()
        self.test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
        self.test_cindex_metric = CindexMetric()
        self.test_ctd_metric = CindexTdMetric()
        self.test_ibs_metric = IbsMetric(event_times)

        self.train_loss_scores, self.train_ci_scores = list(), list()
        self.train_ctd_scores, self.train_ibs_scores = list(), list()
        self.valid_loss_scores, self.valid_ci_scores = list(), list()
        self.test_loss_scores, self.test_ci_scores = list(), list()
        self.test_ctd_scores, self.test_ibs_scores = list(), list()

        self.train_times, self.test_times = list(), list()

    def train_and_evaluate(self):
        for epoch in range(self.num_epochs):
            self.train(epoch)
            if self.valid_ds is not None:
                self.validate()
            if self.test_ds is not None:
                self.test()
            self.cleanup()

    def train(self, epoch):
        train_start_time = time()
        print(f"Training epoch {epoch+1}/{self.num_epochs} for {self.model_type}")
        for x, y in self.train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                if self.model_type == "VI":
                    cox_loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                    loss = cox_loss + tf.reduce_mean(self.model.losses) # CoxPHLoss + KL-divergence
                    self.train_loss_metric.update_state(cox_loss)
                else:
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                    self.train_loss_metric.update_state(loss)

                self.train_cindex_metric.update_state(y, logits)

                y_train = convert_to_structured(y["label_time"], y["label_event"])
                self.test_ctd_metric.update_train_state(y_train) # test CTD

                # CTD
                y_train = convert_to_structured(y["label_time"], y["label_event"])
                self.train_ctd_metric.update_train_state(y_train) # train CTD
                self.train_ctd_metric.update_test_state(y_train)
                self.train_ctd_metric.update_pred_state(logits)

                # IBS
                self.train_ibs_metric.update_train_state(y_train) # train IBS
                self.train_ibs_metric.update_train_pred(logits)
                self.test_ibs_metric.update_train_state(y_train)
                self.test_ibs_metric.update_train_pred(logits)

            with tf.name_scope("gradients"):
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        print(f"Completed epoch {epoch+1}/{self.num_epochs} for {self.model_type}")
        total_train_time = time() - train_start_time

        epoch_loss = self.train_loss_metric.result()
        epoch_ci = self.train_cindex_metric.result()['cindex']
        epoch_ctd = self.train_ctd_metric.result()['cindex']
        epoch_ibs = self.train_ibs_metric.result()

        self.train_loss_scores.append(float(epoch_loss))
        self.train_ci_scores.append(float(epoch_ci))
        self.train_ctd_scores.append(float(epoch_ctd))
        self.train_ibs_scores.append(float(epoch_ibs))

        self.train_times.append(float(total_train_time))

    def validate(self):
        for x, y in self.valid_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            logits = self.model(x, training=False)
            loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
            self.valid_loss_metric.update_state(loss)
            self.valid_cindex_metric.update_state(y, logits)

        epoch_valid_loss = self.valid_loss_metric.result()
        epoch_valid_ci = self.valid_cindex_metric.result()['cindex']
        self.valid_loss_scores.append(float(epoch_valid_loss))
        self.valid_ci_scores.append(float(epoch_valid_ci))

    def cleanup(self):
        self.train_loss_metric.reset_states()
        self.train_cindex_metric.reset_states()
        self.train_ctd_metric.reset_states()
        self.train_ibs_metric.reset_states()
        self.valid_loss_metric.reset_states()
        self.valid_cindex_metric.reset_states()
        self.test_loss_metric.reset_states()
        self.test_cindex_metric.reset_states()
        self.test_ctd_metric.reset_states()
        self.test_ibs_metric.reset_states()

    def test(self):
        test_start_time = time()
        for x, y in self.test_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            if self.model_type == "MCD" or self.model_type == "VI":
                runs = 100
                logits_cpd = np.zeros((runs, len(x)), dtype=np.float32)
                for i in range(0, runs):
                    logits_cpd[i,:] = np.reshape(self.model(x, training=False).sample(), len(x))
                logits = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
            else:
                logits = self.model(x, training=False)
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)

            self.test_loss_metric.update_state(loss)
            self.test_cindex_metric.update_state(y, logits)

            # CTD
            y_test = convert_to_structured(y["label_time"], y["label_event"])
            self.test_ctd_metric.update_test_state(y_test)
            self.test_ctd_metric.update_pred_state(logits)

            # IBS
            self.test_ibs_metric.update_test_state(y_test)
            self.test_ibs_metric.update_test_pred(logits)

        total_test_time = time() - test_start_time

        epoch_loss = self.test_loss_metric.result()
        epoch_ci = self.test_cindex_metric.result()['cindex']
        epoch_ctd = self.test_ctd_metric.result()['cindex']
        epoch_ibs = self.test_ibs_metric.result()

        self.test_loss_scores.append(float(epoch_loss))
        self.test_ci_scores.append(float(epoch_ci))
        self.test_ctd_scores.append(float(epoch_ctd))
        self.test_ibs_scores.append(float(epoch_ibs))
        self.test_times.append(float(total_test_time))
