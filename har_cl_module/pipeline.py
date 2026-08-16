"""
pipeline.py
===========
Módulo generalizado e integrado que une modelos, dados, scaler, benchmark e
a fábrica de estratégias em uma única função de execução (`run_experiment`),
parametrizável por:

  - strategy_name : uma das 12 estratégias em `strategies.STRATEGIES`
  - modality      : 'tabular' | 'timeseries'
  - scenario      : 'class_il' | 'domain_il'
  - quick         : usa dados sintéticos pequenos + 1 época (smoke test)

Isso substitui os ~12 scripts quase-duplicados do repositório original por
um único ponto de entrada parametrizado.
"""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn, optim

from avalanche.training.plugins import EvaluationPlugin

from .models import SimpleMLPNorm, SimpleSequenceClassifierNorm, GradientClippingPlugin, CN8, replace_bn
from .scaler import IncrementalScalerTransform, UpdateScalerPlugin
from .metrics import f1_metrics
from .data import RealHARDataset, make_class_il_data, make_domain_il_data
from .benchmark import build_class_il_benchmark, build_domain_il_benchmark
from .strategies import build_strategy, FixedParams, default_quick_params, STRATEGIES

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class ExperimentConfig:
    strategy_name: str
    modality: str = "tabular"          # 'tabular' | 'timeseries'
    scenario: str = "class_il"         # 'class_il' | 'domain_il'
    n_classes: int = 4
    n_experiences: int = 3             # class_il: nº de experiências (split de classes)
    n_train_users: int = 3             # domain_il: nº de experiências (1 por usuário)
    samples_per_class: int = 12
    seed: int = 42
    use_cn: bool = True                # aplica Continual Normalization (CN8) no modelo
    device: Optional[str] = None
    # shape específico p/ timeseries
    seq_len: int = 50
    n_channels: int = 6
    # tabular
    input_size: int = 64
    hidden_size: int = 64
    # fonte dos dados
    data_source: Literal["synthetic", "real"] = "synthetic"
    data_root: Optional[str] = None
    train_users: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    test_users: tuple[int, ...] = (7, 8)
    validation_user: Optional[int] = None
    max_samples_per_activity: Optional[int] = None


def _build_model(cfg: ExperimentConfig):
    if cfg.modality == "tabular":
        model = SimpleMLPNorm(num_classes=cfg.n_classes, input_size=cfg.input_size,
                               hidden_size=cfg.hidden_size)
        if cfg.use_cn:
            replace_bn(model, "model", CN8)
    else:
        model = SimpleSequenceClassifierNorm(
            input_size=cfg.n_channels, hidden_size=32, n_classes=cfg.n_classes,
            rnn_layers=1, drop_rate=0.1, proj_size=None, subsample_factor=2,
        )
        # SimpleSequenceClassifierNorm usa LayerNorm nativamente (sem BatchNorm1d)
    return model


def _build_benchmark(cfg: ExperimentConfig):
    if cfg.data_source == "real":
        if not cfg.data_root:
            raise ValueError("data_root é obrigatório quando data_source='real'.")
        if cfg.scenario == "class_il":
            train_ds = RealHARDataset(
                cfg.data_root, cfg.modality, cfg.train_users, split="train",
                max_per_activity=cfg.max_samples_per_activity, seed=cfg.seed,
            )
            test_ds = RealHARDataset(
                cfg.data_root, cfg.modality, cfg.test_users, split="all",
                max_per_activity=cfg.max_samples_per_activity, seed=cfg.seed,
            )
            _apply_real_shape(cfg, train_ds, test_ds)
            benchmark = build_class_il_benchmark(
                train_ds, test_ds, n_experiences=cfg.n_experiences, seed=cfg.seed
            )
            return benchmark, "test_stream"

        train_users = tuple(cfg.train_users[:cfg.n_train_users])
        if len(train_users) != cfg.n_train_users:
            raise ValueError("n_train_users é maior que a quantidade de train_users fornecida.")
        validation_user = cfg.validation_user
        if validation_user is None:
            candidates = [user for user in cfg.train_users if user not in train_users]
            if not candidates:
                raise ValueError("Informe validation_user ou mantenha ao menos um usuário fora de train_users.")
            validation_user = candidates[0]
        train_dss = [
            RealHARDataset(cfg.data_root, cfg.modality, [user], split="train",
                           max_per_activity=cfg.max_samples_per_activity, seed=cfg.seed)
            for user in train_users
        ]
        valid_ds = RealHARDataset(cfg.data_root, cfg.modality, [validation_user], split="val",
                                  max_per_activity=cfg.max_samples_per_activity, seed=cfg.seed)
        test_ds = RealHARDataset(cfg.data_root, cfg.modality, cfg.test_users, split="all",
                                 max_per_activity=cfg.max_samples_per_activity, seed=cfg.seed)
        _apply_real_shape(cfg, *train_dss, valid_ds, test_ds)
        return build_domain_il_benchmark(train_dss, valid_ds, test_ds), "valid_stream"

    shape_kwargs = {}
    if cfg.modality == "timeseries":
        shape_kwargs = {"seq_len": cfg.seq_len, "n_channels": cfg.n_channels}
    else:
        shape_kwargs = {"input_size": cfg.input_size}

    if cfg.scenario == "class_il":
        train_ds, test_ds = make_class_il_data(
            cfg.modality, cfg.n_classes, cfg.samples_per_class, seed=cfg.seed, **shape_kwargs
        )
        benchmark = build_class_il_benchmark(
            train_ds, test_ds, n_experiences=cfg.n_experiences, seed=cfg.seed
        )
        stream_name = "test_stream"
    else:
        train_dss, valid_ds, test_ds = make_domain_il_data(
            cfg.modality, cfg.n_train_users, cfg.n_classes, cfg.samples_per_class,
            seed=cfg.seed, **shape_kwargs
        )
        benchmark = build_domain_il_benchmark(train_dss, valid_ds, test_ds)
        stream_name = "valid_stream"
    return benchmark, stream_name


def _apply_real_shape(cfg: ExperimentConfig, *datasets):
    """Ajusta a arquitetura aos dados carregados e valida os rótulos."""
    classes = sorted({int(label) for dataset in datasets for label in dataset.targets.tolist()})
    if classes != list(range(max(classes) + 1)):
        raise ValueError(f"Os rótulos devem ser consecutivos a partir de 0; recebidos: {classes}")
    cfg.n_classes = len(classes)
    shape = datasets[0].data.shape
    if cfg.modality == "tabular":
        cfg.input_size = int(shape[1])
    else:
        cfg.seq_len, cfg.n_channels = int(shape[1]), int(shape[2])


def _extract_f1_per_experience(result: dict, stream_name: str, n_experiences: int):
    """Busca robusta pela chave F1 por experiência, tolerante à presença/
    ausência de 'TaskXXX' na chave (formatos variaram entre os scripts
    originais Class-IL vs Domain-IL)."""
    pattern = re.compile(rf"F1_Exp_macro/eval_phase/{stream_name}/.*Exp(\d+)$")
    per_exp = {}
    for key, value in result.items():
        m = pattern.match(key)
        if m:
            per_exp[int(m.group(1))] = value
    return [per_exp[i] for i in sorted(per_exp) if i < n_experiences]


def run_experiment(cfg: ExperimentConfig, hparams: Optional[dict] = None):
    """
    Executa um experimento completo (treino incremental + avaliação) para
    UMA estratégia/cenário/modalidade, retornando um dicionário com F1 por
    experiência, F1 médio final e tempo de execução.

    `hparams`: dicionário fixo de hiperparâmetros (usa `default_quick_params`
    se None) — em produção, substitua por um `optuna.trial.Trial` real
    passado via `trial=` (a fábrica `build_strategy` aceita ambos).
    """
    t0 = time.time()
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)

    benchmark, stream_name = _build_benchmark(cfg)
    model = _build_model(cfg).to(device)

    params = dict(default_quick_params(cfg.strategy_name))
    if hparams:
        params.update(hparams)
    trial = FixedParams(params)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(f1_metrics(epoch=True, experience=True, stream=True,
                                               per_task=False, average="macro"))
    clipping_plugin = GradientClippingPlugin(clip_value=1.0)
    scaler_t = IncrementalScalerTransform()
    scaler_plugin = UpdateScalerPlugin(scaler_t)

    strategy = build_strategy(
        name=cfg.strategy_name, model=model, optimizer=optimizer, criterion=criterion,
        train_epochs=params["train_epochs"], device=device, eval_plugin=eval_plugin,
        clipping_plugin=clipping_plugin, scaler_plugin=scaler_plugin, trial=trial,
        train_mb_size=8, eval_mb_size=8,
    )

    eval_stream = benchmark.test_stream if stream_name == "test_stream" else benchmark.valid_stream

    results = []
    if cfg.strategy_name == "Joint":
        strategy.train(benchmark.train_stream)
        results.append(strategy.eval(eval_stream))
    else:
        for exp in benchmark.train_stream:
            strategy.train(exp)
            results.append(strategy.eval(eval_stream[:exp.current_experience + 1]))

    final_result = results[-1]
    f1_per_exp = _extract_f1_per_experience(final_result, stream_name, cfg.n_experiences)
    final_f1 = float(np.mean(f1_per_exp)) if f1_per_exp else 0.0

    elapsed = time.time() - t0
    return {
        "strategy": cfg.strategy_name,
        "scenario": cfg.scenario,
        "modality": cfg.modality,
        "f1_per_experience": f1_per_exp,
        "final_f1": final_f1,
        "elapsed_sec": elapsed,
        "n_experiences_evaluated": len(f1_per_exp),
    }
