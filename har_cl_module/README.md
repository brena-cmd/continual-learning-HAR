# har_cl_module — módulo generalizado e integrado de CL para HAR
Estratégias disponíveis:

```
Naive, Replay, EWC, LWF, AGEM, GEM, DER,
ReplayLWF (replay-LWF), LWFReplay (LWF-replay),
MIR, Cumulative (upper bound CL), Joint (upper bound real)
```

## Rodando o teste rápido

```bash
python -m har_cl_module.quick_test
```

## Usando com os dados REAIS (PAMAP2)
Informe a raiz que contém `catch22_flat_subs/` e `novelty_human_activity/`:

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
