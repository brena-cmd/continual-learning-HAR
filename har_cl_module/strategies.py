"""
strategies.py
=============
Fábrica ÚNICA e generalizada de estratégias de Continual Learning, cobrindo
exatamente as 12 estratégias que permanecem no escopo da dissertação (iCaRL
foi removido):

    Naive, Replay, EWC, LWF, AGEM, GEM, DER,
    replay-LWF   (-> 'ReplayLWF': Replay como strategy base + LwFPlugin)
    LWF-replay   (-> 'LWFReplay': LwF como strategy base + ReplayPlugin)
    MIR, Cumulative (upper bound CL), Joint (upper bound offline/real)

Consolida em uma função só a lógica que antes estava duplicada e levemente
divergente entre `class-il-tabular/*.py`, `class-il-timeseries/*.py`,
`domain-il-tabular/*.py` e `domain-il-timeseries/*.py`.
"""
from __future__ import annotations

from avalanche.training import (
    Naive, Replay, EWC, LwF, AGEM, GEM, DER, Cumulative, JointTraining,
)
from avalanche.training.plugins import LwFPlugin, ReplayPlugin, EWCPlugin
from avalanche.training.plugins.mir import MIRPlugin
import torch


class SafeEWCPlugin(EWCPlugin):
    """
    `EWCPlugin` do Avalanche com correção para o erro
    `RuntimeError: cudnn RNN backward can only be called in training mode`.

    CAUSA RAIZ (avalanche 0.6.0): `compute_importances` chama `model.eval()`
    e depois `loss.backward()` para acumular as importâncias de Fisher. Isso
    é inofensivo para MLPs, mas quebra com cuDNN em modelos com LSTM/GRU/RNN
    (nosso `SimpleSequenceClassifierNorm`) em GPU, pois o cuDNN só salva o
    "reserve space" necessário pro backward quando o forward roda em modo
    treino.

    O próprio Avalanche tenta se proteger disso:
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.train()
    só que `strategy.device` é um `torch.device` (ex.: `torch.device('cuda:0')`),
    e `torch.device('cuda:0') == "cuda"` é `False` — então essa proteção
    NUNCA dispara no caso comum de `device = torch.device("cuda" if
    torch.cuda.is_available() else "cpu")`. Confirmado empiricamente.

    IMPORTANTE: subclassar a strategy `avalanche.training.EWC` e sobrescrever
    `compute_importances` (como no script original da autora,
    `class-il-timeseries/EWC-class-il-all-strategies-std-inc.py`) NÃO
    funciona nesta versão do Avalanche, porque `EWC` (a strategy) cria seu
    PRÓPRIO `EWCPlugin` internamente e é esse plugin — não a strategy — quem
    de fato roda `compute_importances`. Por isso construímos EWC aqui como
    `Naive` + este plugin customizado, em vez de usar a classe `EWC`.
    """

    def compute_importances(self, model, criterion, optimizer, dataset, device,
                             batch_size, num_workers=0):
        import warnings
        from torch.utils.data import DataLoader
        from avalanche.models.utils import avalanche_forward
        from avalanche.training.utils import zerolike_params_dict

        model.eval()

        # Correção: comparação robusta de device (funciona com
        # torch.device('cuda'), torch.device('cuda:0') e a string "cuda").
        device_is_cuda = torch.device(device).type == "cuda"
        if device_is_cuda:
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.train()

        importances = zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                                 num_workers=num_workers)
        for batch in dataloader:
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances.items()):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()
        return importances


STRATEGIES = [
    "Naive", "Replay", "EWC", "LWF", "AGEM", "GEM", "DER",
    "ReplayLWF", "LWFReplay", "MIR", "Cumulative", "Joint",
]

# Estratégias que precisam de tratamento especial no loop de treino
# (Joint treina no stream inteiro de uma vez, não experiência-a-experiência).
JOINT_LIKE_STRATEGIES = {"Joint"}


class FixedParams:
    """
    Adaptador com a MESMA interface de um `optuna.trial.Trial`
    (`suggest_float`, `suggest_int`, `suggest_categorical`), mas que sempre
    devolve um valor fixo de um dicionário. Permite reusar `build_strategy`
    tanto durante a otimização Optuna quanto no teste rápido / avaliação
    final com melhores hiperparâmetros (mesmo padrão do `BestTrial` usado
    nos scripts originais).
    """

    def __init__(self, params: dict):
        self.params = params

    def suggest_float(self, name, *a, **kw):
        return self.params[name]

    def suggest_int(self, name, *a, **kw):
        return self.params[name]

    def suggest_categorical(self, name, *a, **kw):
        return self.params[name]


def default_quick_params(strategy_name: str, small_mem: int = 20) -> dict:
    """Hiperparâmetros mínimos e baratos, usados apenas no teste rápido/smoke
    test (não são hiperparâmetros otimizados — servem só para exercitar o
    pipeline de ponta a ponta com custo computacional mínimo)."""
    p = {
        "lr": 1e-3, "weight_decay": 1e-5, "train_epochs": 1,
        "mem_size": small_mem,
        "ewc_lambda": 10.0, "ewc_mode": "online", "decay_factor": 0.5,
        "lwf_alpha": 1.0, "lwf_temperature": 2,
        "agem_patterns_per_exp": small_mem, "agem_sample_size": 16,
        "gem_patterns_per_exp": small_mem, "gem_memory_strength": 0.5,
        "der_mem_size": small_mem, "der_alpha": 0.5, "der_beta": 0.5,
        "mir_mem_size": small_mem, "mir_batch_size_mem": 8, "mir_subsample": small_mem,
    }
    return p


def build_strategy(name: str, model, optimizer, criterion, train_epochs, device,
                    eval_plugin, clipping_plugin, scaler_plugin, trial,
                    train_mb_size: int = 32, eval_mb_size: int = 32, eval_every: int = 0):
    """
    Fábrica única de estratégias. `trial` pode ser um `optuna.trial.Trial`
    real (durante a busca de hiperparâmetros) ou um `FixedParams` (avaliação
    final / teste rápido) — a interface usada (`suggest_*`) é idêntica.

    Ordem dos plugins é sempre [clipping, (plugins específicos da estratégia),
    scaler] — o scaler deve ser o último, pois normaliza o mini-batch
    imediatamente antes do forward (ver `scaler.py`).
    """
    common = dict(
        model=model, optimizer=optimizer, criterion=criterion,
        train_mb_size=train_mb_size, train_epochs=train_epochs,
        eval_mb_size=eval_mb_size, eval_every=eval_every,
        device=device, evaluator=eval_plugin,
    )

    if name == "Naive":
        return Naive(**common, plugins=[clipping_plugin, scaler_plugin])

    if name == "Replay":
        mem_size = trial.suggest_categorical("mem_size", [100, 200, 300])
        return Replay(**common, mem_size=mem_size, plugins=[clipping_plugin, scaler_plugin])

    if name == "EWC":
        ewc_lambda = trial.suggest_float("ewc_lambda", 1.0, 1000.0, log=True)
        mode = trial.suggest_categorical("ewc_mode", ["separate", "online"])
        decay_factor = 0.5 if mode == "online" else None
        # Construído como Naive + SafeEWCPlugin (não via `avalanche.training.EWC`,
        # que instancia seu próprio EWCPlugin problemático internamente — ver
        # docstring de SafeEWCPlugin).
        safe_ewc_plugin = SafeEWCPlugin(ewc_lambda=ewc_lambda, mode=mode,
                                         decay_factor=decay_factor,
                                         keep_importance_data=False)
        return Naive(**common, plugins=[clipping_plugin, safe_ewc_plugin, scaler_plugin])

    if name == "LWF":
        alpha = trial.suggest_float("lwf_alpha", 0.1, 10.0, log=True)
        temperature = trial.suggest_categorical("lwf_temperature", [1, 2, 5, 10])
        return LwF(**common, alpha=alpha, temperature=temperature,
                   plugins=[clipping_plugin, scaler_plugin])

    if name == "AGEM":
        patterns_per_exp = trial.suggest_categorical("agem_patterns_per_exp", [50, 100, 200])
        sample_size = trial.suggest_categorical("agem_sample_size", [64, 128, 256])
        return AGEM(**common, patterns_per_exp=patterns_per_exp, sample_size=sample_size,
                    plugins=[clipping_plugin, scaler_plugin])

    if name == "GEM":
        patterns_per_exp = trial.suggest_categorical("gem_patterns_per_exp", [50, 100, 200])
        memory_strength = trial.suggest_float("gem_memory_strength", 0.0, 1.0)
        return GEM(**common, patterns_per_exp=patterns_per_exp, memory_strength=memory_strength,
                   plugins=[clipping_plugin, scaler_plugin])

    if name == "DER":
        mem_size = trial.suggest_categorical("der_mem_size", [100, 200, 300])
        alpha = trial.suggest_float("der_alpha", 0.1, 1.0)
        beta = trial.suggest_float("der_beta", 0.0, 1.0)
        return DER(**common, mem_size=mem_size, batch_size_mem=train_mb_size,
                   alpha=alpha, beta=beta, plugins=[clipping_plugin, scaler_plugin])

    if name == "ReplayLWF":
        # Replay (strategy base) + LwFPlugin (regularização por distilação)
        mem_size = trial.suggest_categorical("mem_size", [100, 200, 300])
        alpha = trial.suggest_float("lwf_alpha", 0.1, 10.0, log=True)
        temperature = trial.suggest_categorical("lwf_temperature", [1, 2, 5, 10])
        lwf_plugin = LwFPlugin(alpha=alpha, temperature=temperature)
        return Replay(**common, mem_size=mem_size,
                      plugins=[clipping_plugin, lwf_plugin, scaler_plugin])

    if name == "LWFReplay":
        # LwF (strategy base) + ReplayPlugin (buffer auxiliar)
        mem_size = trial.suggest_categorical("mem_size", [100, 200, 300])
        alpha = trial.suggest_float("lwf_alpha", 0.1, 10.0, log=True)
        temperature = trial.suggest_categorical("lwf_temperature", [1, 2, 5, 10])
        replay_plugin = ReplayPlugin(mem_size=mem_size)
        return LwF(**common, alpha=alpha, temperature=temperature,
                   plugins=[clipping_plugin, replay_plugin, scaler_plugin])

    if name == "MIR":
        mem_size = trial.suggest_categorical("mir_mem_size", [100, 200, 300])
        batch_size_mem = trial.suggest_categorical("mir_batch_size_mem", [32, 64])
        subsample = trial.suggest_categorical("mir_subsample", [50, 100, 200])
        mir_plugin = MIRPlugin(mem_size=mem_size, batch_size_mem=batch_size_mem, subsample=subsample)
        return Naive(**common, plugins=[clipping_plugin, mir_plugin, scaler_plugin])

    if name == "Cumulative":
        # Upper bound incremental: retreina em todos os dados vistos até agora.
        return Cumulative(**common, plugins=[clipping_plugin, scaler_plugin])

    if name == "Joint":
        # Upper bound offline: treina uma única vez com o stream completo.
        return JointTraining(**common, plugins=[clipping_plugin, scaler_plugin])

    raise ValueError(
        f"Estratégia desconhecida: {name}. Opções: {', '.join(STRATEGIES)}"
    )
