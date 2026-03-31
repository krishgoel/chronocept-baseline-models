"""
This module implements state-of-the-art baselines with proper distributional regression,
modern encoders, and modular architecture.
"""

from .base import BaseModel, BaseTransformerModel
from .losses import SkewNormalNLL, GaussianNLL, MSELoss
from .pooling import MeanPooling, AttentionPooling, PoolerOutput
from .heads import LinearHead, FFNNHead, MultiTaskHead
from .encoders import SBERTEncoder, RoBERTaEncoder, DeBERTaEncoder, DistilBERTEncoder
from .sbert_models import SBERTFFNN, SBERTBiLSTM
from .transformer_models import RoBERTaRegression, DeBERTaRegression, DistilBERTRegression
from .mtdnn import MTDNNModel
from .legacy import LegacyBERTRegression

__all__ = [
    'BaseModel', 'BaseTransformerModel',
    'SkewNormalNLL', 'GaussianNLL', 'MSELoss',
    'MeanPooling', 'AttentionPooling', 'PoolerOutput',
    'LinearHead', 'FFNNHead', 'MultiTaskHead',
    'SBERTEncoder', 'RoBERTaEncoder', 'DeBERTaEncoder', 'DistilBERTEncoder',
    'SBERTFFNN', 'SBERTBiLSTM',
    'RoBERTaRegression', 'DeBERTaRegression', 'DistilBERTRegression',
    'MTDNNModel', 'LegacyBERTRegression'
]
