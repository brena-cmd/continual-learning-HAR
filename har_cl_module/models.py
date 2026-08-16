"""
models.py
=========
Modelos usados nos experimentos de HAR/CL.

A normalização "Continual Normalization" (CN8/replace_bn) agora usa a
implementação ORIGINAL do repositório da autora (`CN.py`, enviado
posteriormente e copiado para dentro deste pacote) em vez da reconstrução
best-effort da primeira versão deste módulo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils as torch_utils

from avalanche.models.base_model import BaseModel
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .CN import CN4, CN8, CN16, CN32, CN64, replace_bn  # implementação ORIGINAL da autora

__all__ = [
    "GradientClippingPlugin", "SimpleMLPNorm", "SimpleSequenceClassifierNorm",
    "CN4", "CN8", "CN16", "CN32", "CN64", "replace_bn",
]


# --------------------------------------------------------------------------
# Gradient Clipping (idêntico ao usado em todos os scripts originais)
# --------------------------------------------------------------------------
class GradientClippingPlugin(SupervisedPlugin):
    """Aplica clipping de gradiente após o backward, antes do optimizer.step()."""

    def __init__(self, clip_value: float = 1.0):
        super().__init__()
        self.clip_value = clip_value

    def before_update(self, strategy, **kwargs):
        torch_utils.clip_grad_norm_(strategy.model.parameters(), self.clip_value)


# --------------------------------------------------------------------------
# Modelo tabular (catch22 features) — usado em Class-IL / Domain-IL Tabular
# --------------------------------------------------------------------------
class SimpleMLPNorm(nn.Module, BaseModel):
    """MLP com BatchNorm1d (substituível por CN8 via `replace_bn`)."""

    def __init__(self, num_classes=12, input_size=682, hidden_size=464,
                 hidden_layers=1, drop_rate=0.5):
        super().__init__()
        layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=drop_rate),
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(p=drop_rate),
                ),
            )
        self.model = layers  # nome "model" mantido p/ compatibilidade com scripts antigos
        self.features = self.model  # alias usado em alguns scripts (`replace_bn(model, 'features', ...)`)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous().view(x.size(0), self._input_size)
        x = self.model(x)
        return self.classifier(x)

    def get_features(self, x):
        x = x.contiguous().view(x.size(0), self._input_size)
        return self.model(x)


# --------------------------------------------------------------------------
# Modelo de série temporal (janelas brutas multissensor) — Class-IL/Domain-IL TS
# --------------------------------------------------------------------------
class SimpleSequenceClassifierNorm(nn.Module, BaseModel):
    """LSTM com subsample CNN, LayerNorm e inicialização ortogonal."""

    def __init__(self, input_size: int, hidden_size: int = 128, n_classes: int = 12,
                 rnn_layers: int = 2, drop_rate: float = 0.3, proj_size: int = None,
                 batch_first: bool = True, subsample_factor: int = 10):
        super().__init__()
        self.batch_first = batch_first
        self.subsample = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=subsample_factor,
                      stride=subsample_factor, groups=input_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers,
            batch_first=batch_first, dropout=drop_rate if rnn_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=drop_rate)

        if proj_size is not None and proj_size != hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, proj_size),
                nn.ReLU(inplace=True),
                nn.LayerNorm(proj_size),
                nn.Dropout(p=drop_rate),
            )
            classifier_in = proj_size
        else:
            self.projection = None
            classifier_in = hidden_size

        self.classifier = nn.Linear(classifier_in, n_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.subsample(x)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        last = out[:, -1] if self.batch_first else out[-1]
        last = self.norm(last)
        last = self.dropout(last)
        if self.projection is not None:
            last = self.projection(last)
        return self.classifier(last)

    def get_features(self, x):
        x = x.permute(0, 2, 1)
        x = self.subsample(x)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        last = out[:, -1] if self.batch_first else out[-1]
        last = self.norm(last)
        last = self.dropout(last)
        if self.projection is not None:
            last = self.projection(last)
        return last
