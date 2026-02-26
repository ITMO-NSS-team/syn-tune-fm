"""Генеративные модели: каждая модель в своей папке с model.py, наследуется от BaseDataGenerator."""
from src.generators.base import BaseDataGenerator
from src.generators.gaussian import GaussianCopulaGenerator
from src.generators.gmm import GMMGenerator
from src.generators.ctgan import CTGANGenerator
from src.generators.tvae import TVAEGenerator
from src.generators.mixed_model import MixedModelGenerator
from src.generators.table_augmentation import TableAugmentationGenerator

__all__ = [
    "BaseDataGenerator",
    "GaussianCopulaGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "GMMGenerator",
    "MixedModelGenerator",
    "TableAugmentationGenerator",
]
