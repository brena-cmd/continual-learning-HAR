"""
metrics.py
==========
Camada fina de compatibilidade: reexporta a implementação REAL de
`F1Score.py` (arquivo original da autora, enviado posteriormente e copiado
verbatim para dentro deste pacote) — substitui a reconstrução best-effort
usada na primeira versão deste módulo.

Mantido como módulo separado (em vez de importar `F1Score` diretamente em
`pipeline.py`) apenas por estabilidade da API interna do pacote.
"""
from __future__ import annotations

from .F1Score import (
    F1Score,
    TaskAwareF1,
    F1PluginMetric,
    F1PerTaskPluginMetric,
    EpochF1,
    ExperienceF1,
    ExperienceF1PerTask,
    StreamF1,
    f1_metrics,
)
from .MinibatchF1Score import MinibatchF1Score

__all__ = [
    "F1Score", "TaskAwareF1", "F1PluginMetric", "F1PerTaskPluginMetric",
    "EpochF1", "ExperienceF1", "ExperienceF1PerTask", "StreamF1",
    "f1_metrics", "MinibatchF1Score",
]
