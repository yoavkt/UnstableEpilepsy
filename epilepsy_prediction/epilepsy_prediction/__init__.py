"""Epilepsy prediction"""
__version__ = "0.1.0"
__author__ = 'Yoav Kan-Tor'
__credits__ = 'IBM research'
from .utils import data_preprocess, load_model

__all__ = [
    "data_preprocess",
    "load_model",
]
