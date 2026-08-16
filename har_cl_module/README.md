# har_cl_module — módulo generalizado e integrado de CL para HAR

Consolida os ~12 scripts quase-duplicados do repositório (`class-il-tabular/*`,
`class-il-timeseries/*`, `domain-il-tabular/*`, `domain-il-timeseries/*`) em
um único pacote parametrizável, cobrindo as **12 estratégias remanescentes**
(iCaRL foi removido do escopo):

```
Naive, Replay, EWC, LWF, AGEM, GEM, DER,
ReplayLWF (replay-LWF), LWFReplay (LWF-replay),
MIR, Cumulative (upper bound CL), Joint (upper bound real)
```

## Estrutura

| Arquivo                | Papel |
|--------------------------|-------|
| `CN.py`                  | **Original da autora** — `CN4/CN8/CN16/CN32/CN64` + `replace_bn` (Continual Normalization) |
| `F1Score.py`             | **Original da autora** — `F1Score`, `TaskAwareF1`, `EpochF1/ExperienceF1/StreamF1`, `f1_metrics` |
| `MinibatchF1Score.py`    | **Original da autora** — F1 por minibatch (import ajustado p/ relativo, ver nota) |
| `resources/value_to_sequence.pkl` | **Original da autora** — mapa `activity_code -> label 0..11` usado pelos datasets reais |
| `models.py`              | `SimpleMLPNorm` (tabular) e `SimpleSequenceClassifierNorm` (time-series), `GradientClippingPlugin`; importa `CN8`/`replace_bn` de `CN.py` |
| `scaler.py`              | Cópia direta do `ScalerPlugin.py` original (scaler incremental por mini-batch) |
| `metrics.py`             | Camada fina que reexporta `F1Score.py`/`MinibatchF1Score.py` |
| `data.py`                | `RealHARDataset` (PAMAP2 pré-processado) e datasets sintéticos opcionais |
| `benchmark.py`           | `build_class_il_benchmark` (`nc_benchmark`) e `build_domain_il_benchmark` (`benchmark_from_datasets`) |
| `strategies.py`          | Fábrica única `build_strategy(name, ...)` para as 12 estratégias |
| `pipeline.py`            | `ExperimentConfig` + `run_experiment(cfg)`: monta modelo+benchmark+estratégia e roda treino/avaliação |
| `quick_test.py`          | Smoke test: roda as 12 estratégias em Class-IL/Domain-IL Tabular + subconjunto em Time-series |

## Rodando o teste rápido

```bash
python -m har_cl_module.quick_test
```

Resultado no ambiente de validação (com `CN.py`/`F1Score.py`/
`MinibatchF1Score.py` **originais**, já integrados): **30/30 combinações
OK** (12 estratégias × Class-IL Tabular, 12 × Domain-IL Tabular, 6 ×
Class-IL Time-series), usando dados sintéticos pequenos (10–12
amostras/classe, 1 época). `CN8` foi confirmado substituindo de fato as
camadas `BatchNorm1d` do modelo (`replace_bn` ativo por padrão via
`ExperimentConfig.use_cn=True`).

## ✅ Componentes originais integrados (não são mais reconstruções)

Você enviou depois os 3 arquivos que faltavam no `repositorio.zip` original
— `CN.py`, `F1Score.py` e `MinibatchF1Score.py` — e o `value_to_sequence.pkl`
usado pelos datasets reais. Eles foram copiados **verbatim** para dentro do
pacote (única mudança: o import `from F1Score import F1Score` em
`MinibatchF1Score.py`, que era absoluto, virou `from .F1Score import
F1Score` — import relativo, necessário porque o arquivo agora vive dentro do
pacote `har_cl_module`; nenhuma lógica foi alterada). A reconstrução
best-effort da primeira versão deste módulo foi **descartada**.

Duas observações sobre o `CN.py` original, para sua ciência:
- `_CN.forward` chama `F.batch_norm(..., self.training, momentum, eps)` sem
  proteção para `batch_size == 1` em modo treino — isso pode lançar erro do
  PyTorch ("Expected more than 1 value per channel...") se algum mini-batch
  final tiver exatamente 1 amostra. As configurações do `quick_test.py` já
  evitam isso (tamanhos de dataset múltiplos do `train_mb_size`), mas ao
  usar dados reais com tamanhos de experiência não-múltiplos de
  `train_mb_size`, vale considerar `drop_last=True` no DataLoader ou ajustar
  `train_mb_size`.
- `replace_bn` só substitui camadas exatamente do tipo `torch.nn.BatchNorm1d`
  (checagem por `type(...) == ...`, não `isinstance`) — `SimpleMLPNorm` usa
  `nn.BatchNorm1d` puro, então a substituição funciona normalmente.

## Usando com os dados REAIS (PAMAP2)

O módulo já contém um carregador real, sem caminhos absolutos. Informe a raiz
que contém `catch22_flat_subs/` e `novelty_human_activity/`:

```bash
python -m har_cl_module.quick_test --data-root /caminho/para/data
```

No modo real, o quicktest preserva os splits cronológicos por atividade e o
protocolo por usuário, mas limita cada atividade a duas janelas para continuar
rápido. Para um experimento completo, use `ExperimentConfig` com
`data_source="real"`, `data_root=...` e `max_samples_per_activity=None`.
As dimensões da entrada e as 12 classes são inferidas do dado carregado.

Depois, para reproduzir a otimização de hiperparâmetros completa (100 trials
Optuna, 10 seeds de usuários), basta trocar `FixedParams` por um
`optuna.trial.Trial` real dentro de um `study.optimize(objective, ...)` — a
assinatura de `build_strategy` já aceita ambos sem alteração (é exatamente o
padrão `BestTrial`/`trial` usado nos scripts originais).
