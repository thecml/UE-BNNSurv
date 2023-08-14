# Uncertainty Estimation in Deep Bayesian Survival Models

This repository is the official TensorFlow implementation of "Uncertainty Estimation in Deep Bayesian Survival Models", BHI 2023.

The proposed method is implemented based on the TensorFlow Probability package: https://github.com/tensorflow/probability

Full paper will be available soon on IEEE Xplore.

<p align="center"><img src="https://github.com/thecml/UE-BNNSurv/img/BNN.png" width="95%" height="95%">

In this work we introduce the use of Bayesian inference techniques for survival analysis in neural networks that rely on the Cox’s proportional hazard assumption, for which we discuss a new flexible and effective architecture. We implement three architectures: a fully-deterministic neural network that acts as a baseline, a Bayesian model using variational inference and one using Monte-Carlo Dropout.

Experiments show that the Bayesian models improve predictive performance over SOTA neural networks in a test dataset with few samples (WHAS500, 500 samples) and provide comparable performance in two larger ones (SEER and SUPPORT, 4024 and 8873 samples, respectively)

<p align="center"><img src="https://github.com/thecml/UE-BNNSurv/img/seer_surv_all_models.png" width="40%" height="40%" /> <img src="https://github.com/thecml/UE-BNNSurv/img/seer_surv_grade_mcd.png" width="35%" height="35%" /> <img src="https://github.com/thecml/UE-BNNSurv/img/seer_surv_pdf.png" width="35%" height="35%" />


License
--------
To view the license for this work, visit https://github.com/thecml/UE-BNNSurv/blob/main/LICENSE


Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/UE-BNNSurv/blob/main/requirements.txt).

Code was tested in virtual environment with `Python 3.8`, `TensorFlow 2.11` and `TensorFlow Probability 0.19`


Training
--------
- Make directories `mkdir results` and `mkdir models`.

- Please refer to `paths.py` to set appropriate paths. By default, results are in `results` and models in `models`

- Network configuration using best hyperparameters are found in `configs/*`

- Run the training code:

```
# SOTA models
python train_sota_models.py

# BNN Models
python train_bnn_models.py
```


Evaluation
--------
- After model training, view the results in the `results` folder.


Visualization
---------
- Run the visualization notebooks:
```
jupyter notebook plot_survival_curves.ipynb
jupyter notebook plot_survival_time.ipynb
```


Citation
--------
TBA