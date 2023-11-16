# Uncertainty Estimation in Deep Bayesian Survival Models

This repository is the official TensorFlow implementation of [Uncertainty Estimation in Deep Bayesian Survival Models](https://bhiconference.github.io/BHI2023/2023/pdfs/1570918354.pdf), BHI 2023.

The proposed method is implemented based on [TensorFlow Probability](https://github.com/tensorflow/probability).

<b>Full paper will be published on IEEE Xplore as part of BHI 2023 proceedings. ETA October 2023</b>

<p align="left"><img src="https://github.com/thecml/UE-BNNSurv/blob/main/img/BNN.png" width="40%" height="40%">

In this work, we introduce the use of Bayesian inference techniques for survival analysis in neural networks that rely on the Cox’s proportional hazard assumption, for which we discuss a new flexible and effective architecture. We implement three architectures: a fully-deterministic neural network that acts as a baseline, a Bayesian model using variational inference and one using Monte-Carlo Dropout.

Experiments show that the Bayesian models improve predictive performance over SOTA neural networks in a test dataset with few samples (WHAS500, 500 samples) and provide comparable performance in two larger ones (SEER and SUPPORT, 4024 and 8873 samples, respectively)

<p align="center"><img src="https://github.com/thecml/UE-BNNSurv/blob/main/img/seer_surv_all_models.png" width="30%" height="30%" /> <img src="https://github.com/thecml/UE-BNNSurv/blob/main/img/seer_surv_grade_mcd.png" width="30%" height="30%" /> <img src="https://github.com/thecml/UE-BNNSurv/blob/main/img/seer_surv_pdf.png" width="31%" height="31%" />


License
--------
To view the license for this work, visit https://github.com/thecml/UE-BNNSurv/blob/main/LICENSE


Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/UE-BNNSurv/blob/main/requirements.txt).

Install auton-survival manually from Git:
```
pip install git+https://github.com/autonlab/auton-survival.git
```
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
```
@inproceedings{lillelund_uncertainty_2023,
  author     = {Christian Marius Lillelund and Martin Magris and Christian Fischer Pedersen},
  title      = {Uncertainty Estimation in Deep Bayesian Survival Models},
  booktitle  = {2023 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI) (IEEE BHI 2023)},
  pages      = {18},
  year       = {2023},
}
```
