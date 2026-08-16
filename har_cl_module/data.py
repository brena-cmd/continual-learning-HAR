"""
data.py
=======
O carregador real recebe a raiz dos dados explicitamente e preserva a lógica
dos loaders originais: características catch22 para a representação tabular,
um CSV por segmento para séries temporais e divisão cronológica por atividade.
Os dados sintéticos permanecem disponíveis apenas para um smoke test isolado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Mapa real activity_code -> label_sequencial (0..11) usado pelos datasets
# reais do repositório (`joblib.load('../value_to_sequence.pkl')` dentro de
# `CustomHARDatasetSTD`). Enviado pela autora; incluído aqui em
# `resources/value_to_sequence.pkl` para uso ao trocar os dados sintéticos
# pelos dados reais do PAMAP2/WISDM (ver README, seção "Usando com os dados
# REAIS").
VALUE_TO_SEQUENCE_PATH = Path(__file__).with_name("resources") / "value_to_sequence.pkl"


def load_value_to_sequence() -> dict:
    """Carrega o mapeamento activity_code -> label 0..11 usado nos dados reais."""
    return joblib.load(VALUE_TO_SEQUENCE_PATH)


Split = Literal["train", "val", "test", "all"]


class RealHARDataset(Dataset):
    """Dataset PAMAP2 já segmentado no layout usado nos scripts originais.

    ``data_root`` deve conter, por exemplo::

        data_root/
          catch22_flat_subs/catch22_flat_sub101.csv
          novelty_human_activity/101/y.csv
          novelty_human_activity/101/<um CSV por segmento>.csv

    Para ``tabular``, são lidas as features catch22. Para ``timeseries``,
    cada CSV de segmento é uma amostra (amostras, tempo, canais).
    """

    def __init__(
        self,
        data_root: str | Path,
        modality: Literal["tabular", "timeseries"],
        user_ids: Iterable[int],
        split: Split = "all",
        max_per_activity: int | None = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modality = modality
        self.user_ids = tuple(int(user) for user in user_ids)
        if not self.user_ids:
            raise ValueError("user_ids não pode ser vazio.")
        mapping = load_value_to_sequence()
        xs, ys = [], []

        for user_id in self.user_ids:
            labels_path = self.data_root / "novelty_human_activity" / f"10{user_id}" / "y.csv"
            if not labels_path.is_file():
                raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {labels_path}")
            labels = pd.read_csv(labels_path)[" activity"].map(mapping)
            if labels.isna().any():
                unknown = pd.read_csv(labels_path).loc[labels.isna(), " activity"].unique().tolist()
                raise ValueError(f"Atividades sem rótulo no value_to_sequence: {unknown}")

            features = self._load_user_features(user_id)
            if len(features) != len(labels):
                raise ValueError(
                    f"Usuário {user_id}: {len(features)} amostras em X, mas {len(labels)} em y. "
                    "Confira os CSVs de segmentos e sua ordenação."
                )
            indices = self._select_indices(labels.to_numpy(dtype=np.int64), split, max_per_activity, seed + user_id)
            xs.append(features[indices])
            ys.append(labels.to_numpy(dtype=np.int64)[indices])

        self.data = torch.as_tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
        self.targets = torch.as_tensor(np.concatenate(ys, axis=0), dtype=torch.long)

    def _load_user_features(self, user_id: int) -> np.ndarray:
        if self.modality == "tabular":
            path = self.data_root / "catch22_flat_subs" / f"catch22_flat_sub10{user_id}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo catch22 não encontrado: {path}")
            frame = pd.read_csv(path).ffill().bfill().fillna(0)
            return frame.to_numpy(dtype=np.float32)

        folder = self.data_root / "novelty_human_activity" / f"10{user_id}"
        paths = sorted(path for path in folder.glob("*.csv") if path.name != "y.csv")
        if not paths:
            raise FileNotFoundError(f"Nenhum CSV de série temporal encontrado: {folder}")
        segments = [pd.read_csv(path).ffill().bfill().fillna(0).to_numpy(dtype=np.float32) for path in paths]
        shapes = {segment.shape for segment in segments}
        if len(shapes) != 1:
            raise ValueError(f"Usuário {user_id}: segmentos com formatos diferentes: {sorted(shapes)}")
        return np.stack(segments)

    @staticmethod
    def _select_indices(labels: np.ndarray, split: Split, max_per_activity: int | None, seed: int) -> np.ndarray:
        selected = []
        rng = np.random.default_rng(seed)
        for activity in np.unique(labels):
            activity_indices = np.flatnonzero(labels == activity)
            n_train = round(len(activity_indices) * 0.70)
            n_val = round((len(activity_indices) - n_train) * (2 / 3))
            if split == "train":
                indices = activity_indices[:n_train]
            elif split == "val":
                indices = activity_indices[n_train:n_train + n_val]
            elif split == "test":
                indices = activity_indices[n_train + n_val:]
            else:
                indices = activity_indices
            if max_per_activity is not None and len(indices) > max_per_activity:
                # Subamostragem determinística; não altera a definição da divisão.
                indices = np.sort(rng.choice(indices, size=max_per_activity, replace=False))
            selected.extend(indices.tolist())
        return np.asarray(sorted(selected), dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SyntheticTabularDataset(Dataset):
    """Equivalente sintético ao `CustomHARDatasetSTD` tabular (features catch22)."""

    def __init__(self, n_samples_per_class: int, n_classes: int, input_size: int,
                 seed: int = 42, class_offset: int = 0, noise: float = 1.0,
                 return_task_label: bool = True, centers_seed: int = 0):
        # Os CENTROIDES de cada classe usam uma seed fixa e independente de
        # `seed`/`class_offset`, para que treino e teste (que usam seeds
        # diferentes só para variar a amostragem/ruído) compartilhem a MESMA
        # definição de classe — senão o "conceito" a aprender mudaria entre
        # train e test, tornando a tarefa não-aprendível por construção.
        centers_rng = np.random.default_rng(centers_seed)
        max_classes = 64
        all_centers = centers_rng.normal(scale=3.0, size=(max_classes, input_size))
        rng = np.random.default_rng(seed)
        centers = all_centers[class_offset:class_offset + n_classes]
        X, y = [], []
        for c in range(n_classes):
            samples = centers[c] + rng.normal(scale=noise, size=(n_samples_per_class, input_size))
            X.append(samples)
            y.extend([c + class_offset] * n_samples_per_class)
        X = np.vstack(X).astype("float32")
        order = rng.permutation(len(X))
        self.data = torch.tensor(X[order], dtype=torch.float32)
        self.targets = torch.tensor(np.array(y)[order], dtype=torch.long)
        self.targets_task_labels = torch.zeros(len(self.targets), dtype=torch.long)
        self.return_task_label = return_task_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_task_label:
            return self.data[idx], self.targets[idx], self.targets_task_labels[idx]
        # Class-IL (nc_benchmark) espera um dataset "flat" (x, y) — o rótulo
        # de tarefa é derivado automaticamente pelo próprio nc_benchmark.
        return self.data[idx], self.targets[idx]


class SyntheticTimeSeriesDataset(Dataset):
    """Equivalente sintético ao `CustomHARDatasetSTD` de séries temporais."""

    def __init__(self, n_samples_per_class: int, n_classes: int, seq_len: int,
                 n_channels: int, seed: int = 42, class_offset: int = 0, noise: float = 1.0,
                 return_task_label: bool = True, centers_seed: int = 0):
        # Ver comentário equivalente em SyntheticTabularDataset: centroides
        # de classe fixos (independentes da seed de amostragem) para que
        # treino e teste compartilhem a mesma definição de classe.
        centers_rng = np.random.default_rng(centers_seed)
        max_classes = 64
        all_centers = centers_rng.normal(scale=2.0, size=(max_classes, n_channels))
        rng = np.random.default_rng(seed)
        centers = all_centers[class_offset:class_offset + n_classes]
        X, y = [], []
        for c in range(n_classes):
            base = centers[c][None, None, :]  # (1,1,C)
            samples = base + rng.normal(scale=noise, size=(n_samples_per_class, seq_len, n_channels))
            X.append(samples)
            y.extend([c + class_offset] * n_samples_per_class)
        X = np.vstack(X).astype("float32")
        order = rng.permutation(len(X))
        self.data = torch.tensor(X[order], dtype=torch.float32)
        self.targets = torch.tensor(np.array(y)[order], dtype=torch.long)
        self.targets_task_labels = torch.zeros(len(self.targets), dtype=torch.long)
        self.return_task_label = return_task_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_task_label:
            return self.data[idx], self.targets[idx], self.targets_task_labels[idx]
        return self.data[idx], self.targets[idx]


def make_class_il_data(modality: str, n_classes: int, samples_per_class: int = 12,
                        seed: int = 42, **shape_kwargs):
    """
    Retorna (train_dataset, test_dataset) "flat" para uso com `nc_benchmark`
    (o benchmark cuida de dividir em n_experiences por classes).
    """
    cls = SyntheticTabularDataset if modality == "tabular" else SyntheticTimeSeriesDataset
    train_ds = cls(samples_per_class, n_classes, seed=seed, return_task_label=False, **shape_kwargs)
    test_ds = cls(max(2, samples_per_class // 3), n_classes, seed=seed + 1000,
                   return_task_label=False, **shape_kwargs)
    return train_ds, test_ds


def make_domain_il_data(modality: str, n_train_users: int, n_classes: int,
                         samples_per_class: int = 10, seed: int = 42, **shape_kwargs):
    """
    Retorna (train_datasets_por_usuario: list[Dataset], valid_dataset, test_dataset)
    para uso com `benchmark_from_datasets` (uma experiência por usuário de treino;
    valid/test correspondem a usuários/domínios totalmente separados,
    replicando o protocolo held-out do repositório original).
    """
    cls = SyntheticTabularDataset if modality == "tabular" else SyntheticTimeSeriesDataset
    # return_task_label=False: o rótulo de tarefa é adicionado uma única vez
    # por `wrap_dataset`/`make_avalanche_dataset` (ver benchmark.py). Deixar o
    # dataset bruto também devolver um 3º valor duplicava o campo de task
    # label e quebrava estratégias que fazem DataLoader direto no buffer
    # (ex.: DER), pois o item passava a ter 4 valores em vez de 3.
    train_datasets = [
        cls(samples_per_class, n_classes, seed=seed + u, return_task_label=False, **shape_kwargs)
        for u in range(n_train_users)
    ]
    # domínio de validação/teste com um leve "shift" (offset de seed maior)
    valid_ds = cls(max(2, samples_per_class // 2), n_classes, seed=seed + 500,
                    return_task_label=False, **shape_kwargs)
    test_ds = cls(max(2, samples_per_class // 2), n_classes, seed=seed + 900,
                   return_task_label=False, **shape_kwargs)
    return train_datasets, valid_ds, test_ds
