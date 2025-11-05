"""
Skin Disease Classifier Package
"""

__version__ = "1.0.0"
__author__ = "ML Lab"
__description__ = "AI-powered skin disease classification with explainability"

from . import model
from . import data_pipeline
from . import utils
from . import gradcam

__all__ = [
    'model',
    'data_pipeline',
    'utils',
    'gradcam'
]
