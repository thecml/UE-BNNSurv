"""
paths.py
====================================
Module to hold paths of files.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
MLP_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mlp')
RSF_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'rsf')
COX_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'cox')
COXNET_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'coxnet')
DSM_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dsm')
DCPH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dcph')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')