import numpy as np

def get_mlp_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64]]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            },
            "weight_decay": {
                "values": [1e-3, 1e-4, 1e-5, None]
            },
            "momentum": {
                "values": [0.99, 0.97, 0.95, 0.9, 0.0]
            },
            "optimizer": {
                "values": ["Adam", "SGD", "RMSprop"]
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [0.25]
            },
            "l2_reg": {
                "values": [None, 0.001, 0.01, 0.1]
            }
        }
    }

def get_rsf_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
            },
        "parameters": {
            "n_estimators": {
                "values": [50, 100, 200, 400, 600, 800, 1000]
            },
            "max_depth": {
                "values": [3, 5, 7]
            },
            "min_samples_split": {
                "values": [float(x) for x in np.linspace(0.1, 0.9, 10, endpoint=True)]
            },
            "min_samples_leaf": {
                "values": [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]
            },
            "max_features": {
                "values": [None, 'auto', 'sqrt', 'log2']
            },
        }
    }

def get_cox_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "n_iter": {
                "values": [50, 100]
            },
            "tol": {
                "values": [1e-1, 1e-5, 1e-9]
            }
        }
    }

def get_coxnet_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "l1_ratio": {
                "values": [0.5, 1.0]
            },
            "alpha_min_ratio": {
                "values": [0.01, 0.1, 0.5, "auto"]
            },
            "n_alphas": {
                "values": [10, 50, 100]
            },
            "normalize": {
                "values": [True, False]
            },
            "tol": {
                "values": [1e-1, 1e-5, 1e-7]
            },
            "max_iter": {
                "values": [100000]
            }

        }
    }

def get_dsm_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64]]
            },
            "n_iter": {
                "values": [50, 100, 200, 500, 1000, 5000, 10000]
            }
        }
    }

def get_dcph_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64]]
            },
            "iters": {
                "values": [50, 100, 200, 500, 1000, 5000, 10000]
            },
            "optimizer": {
                "values": ["Adam", "SGD", "RMSProp"]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            },
        }
    }