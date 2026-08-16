"""
benchmark.py
============
Construção generalizada de benchmarks Avalanche para os 2 cenários usados na
dissertação:

  * Class-IL  -> `nc_benchmark`  (split por classes; N experiências fixas)
  * Domain-IL -> `benchmark_from_datasets` (1 experiência por usuário de
                 treino; validação/teste vêm de usuários/domínios não vistos)

Ambos seguem exatamente a mesma lógica usada nos scripts originais do
repositório (`class-il-*.py` e `domain-il-*.py`).
"""
from __future__ import annotations

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset, TransformGroups
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.flat_data import ConstantSequence


def build_class_il_benchmark(train_dataset, test_dataset, n_experiences: int,
                              seed: int = 42, shuffle: bool = True):
    """Cria um benchmark Class-Incremental a partir de datasets "flat"
    (com atributo `.targets`), exatamente como em `class-il-*.py`."""
    return nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=seed,
        fixed_class_order=None,
        shuffle=shuffle,
    )


def wrap_dataset(data, transform_groups=None):
    """
    Envolve um Dataset padrão do PyTorch (com `.targets`) em um
    AvalancheDataset com `targets` e `targets_task_labels` (task 0 para
    todos — Domain-IL não distingue tasks), replicando `wrap_dataset` dos
    scripts `domain-il-*.py`.
    """
    da_targets = DataAttribute(data.targets, "targets")
    da_task_labels = DataAttribute(
        ConstantSequence(0, len(data)), "targets_task_labels", use_in_getitem=True,
    )
    dataset = make_avalanche_dataset(
        data, data_attributes=[da_targets, da_task_labels], transform_groups=transform_groups,
    )
    dataset.task_set = {0: dataset}
    return dataset


def build_domain_il_benchmark(train_datasets: list, valid_dataset, test_dataset):
    """
    Cria um benchmark Domain-Incremental: uma experiência de treino por
    usuário/domínio; os streams de validação e teste usam usuários held-out
    (replicados uma vez por experiência de treino, como em
    `EWC-stdinc-pamap2.py`).
    """
    tgroups = TransformGroups({"train": None, "eval": None})

    wrapped_train = [wrap_dataset(ds, transform_groups=tgroups) for ds in train_datasets]
    wrapped_valid = [wrap_dataset(valid_dataset, transform_groups=tgroups) for _ in train_datasets]
    wrapped_test = [wrap_dataset(test_dataset, transform_groups=tgroups) for _ in train_datasets]

    return benchmark_from_datasets(
        train=wrapped_train,
        valid=wrapped_valid,
        test=wrapped_test,
    )
