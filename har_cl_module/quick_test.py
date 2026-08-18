"""
quick_test.py
=============
Teste rápido ("smoke test") do módulo generalizado e integrado.

Executa TODAS as
12 estratégias remanescentes em:

  1. Class-IL  Tabular      (cenário mais barato -> roda as 12 estratégias)
  2. Domain-IL Tabular       (valida o benchmark_from_datasets genérico)
  3. Class-IL  Time-series   (valida o modelo LSTM genérico, 2 estratégias)

Uso com dados sintéticos (padrão):
    python -m har_cl_module.quick_test

Uso com os dados reais já pré-processados:
    python -m har_cl_module.quick_test --data-root /caminho/para/data
"""
from __future__ import annotations

import argparse
import sys
import traceback

from .pipeline import ExperimentConfig, run_experiment
from .strategies import STRATEGIES


def _run_suite(scenario: str, modality: str, strategy_names, hparams: dict = None, **cfg_kwargs):
    print(f"\n{'=' * 70}")
    print(f"SUITE: scenario={scenario} | modality={modality}")
    print(f"{'=' * 70}")
    rows = []
    f1s = []
    for name in strategy_names:
        cfg = ExperimentConfig(strategy_name=name, scenario=scenario, modality=modality, **cfg_kwargs)
        try:
            res = run_experiment(cfg, hparams=hparams)
            status = "OK"
            detail = (f"final_f1={res['final_f1']:.3f} | "
                      f"exps_avaliadas={res['n_experiences_evaluated']} | "
                      f"tempo={res['elapsed_sec']:.1f}s")
            f1s.append(res["final_f1"])
        except Exception as e:  # noqa: BLE001 - queremos capturar e reportar qualquer falha
            status = "FALHOU"
            detail = f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=2)
        rows.append((name, status, detail))
        print(f"  [{status:6s}] {name:12s} {detail}")

    # Checagem de sanidade: se TODAS as estratégias derem o F1 idêntico,
    # é sinal de que o treino foi curto/pequeno demais para diferenciá-las
    # (ex.: previsões dominadas pela inicialização) — não deve passar em
    # silêncio como se fosse um "OK" comum.
    if len(f1s) == len(strategy_names) and len(set(round(v, 6) for v in f1s)) == 1:
        print(f"  [AVISO ] Todas as {len(f1s)} estratégias desta suíte deram "
              f"F1 IDÊNTICO ({f1s[0]:.3f}) — indício de treino insuficiente "
              f"para diferenciá-las (não é falha de integração, mas o "
              f"resultado não valida comportamento distinto por estratégia).")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Raiz que contém catch22_flat_subs/ e novelty_human_activity/.")
    parser.add_argument("--max-samples-per-activity", type=int, default=2,
                        help="Limite por atividade no modo real; mantém o quicktest curto (padrão: 2).")
    args = parser.parse_args(argv)
    real = args.data_root is not None
    source_kwargs = {
        "data_source": "real" if real else "synthetic",
        "data_root": args.data_root,
        "max_samples_per_activity": args.max_samples_per_activity if real else None,
    }
    all_rows = []

    # 1) Class-IL Tabular — as 12 estratégias
    all_rows += _run_suite(
        "class_il", "tabular", STRATEGIES,
        n_classes=12 if real else 4, n_experiences=3 if real else 2, samples_per_class=12,
        input_size=682 if real else 32, hidden_size=32, **source_kwargs,
    )

    # 2) Domain-IL Tabular — as 12 estratégias
    all_rows += _run_suite(
        "domain_il", "tabular", STRATEGIES,
        n_classes=12 if real else 4, n_train_users=3, samples_per_class=10,
        input_size=682 if real else 32, hidden_size=32, **source_kwargs,
    )

    # 3) Class-IL Time-series — as 12 estratégias
    # NOTA: o modelo LSTM (mais profundo que o MLP tabular) não converge em
    # apenas 1 época/poucos mini-batches — com o `train_epochs=1` padrão do
    # smoke test as previsões ficavam dominadas pela inicialização e todas
    # as 12 estratégias davam EXATAMENTE o mesmo F1 (sinal de treino
    # insuficiente, não de integração quebrada). Usamos mais épocas aqui
    # para que o teste rápido também sirva para checar que as estratégias
    # de fato aprendem e se diferenciam, não só que "não dão erro".
    all_rows += _run_suite(
        "class_il", "timeseries", STRATEGIES, hparams={"train_epochs": 20},
        n_classes=12 if real else 4, n_experiences=3 if real else 2, samples_per_class=8,
        seq_len=1000 if real else 30, n_channels=31 if real else 4, **source_kwargs,
    )

    # 4) Domain-IL Time-series — as 12 estratégias (mesmo motivo acima)
    all_rows += _run_suite(
        "domain_il", "timeseries", STRATEGIES, hparams={"train_epochs": 20},
        n_classes=12 if real else 4, n_train_users=3, samples_per_class=8,
        seq_len=1000 if real else 30, n_channels=31 if real else 4, **source_kwargs,
    )

    print(f"\n{'=' * 70}")
    print("RESUMO FINAL")
    print(f"{'=' * 70}")
    n_ok = sum(1 for _, status, _ in all_rows if status == "OK")
    n_fail = len(all_rows) - n_ok
    for name, status, detail in all_rows:
        print(f"  [{status:6s}] {name:12s} {detail}")
    print(f"\nTotal: {len(all_rows)} | OK: {n_ok} | FALHOU: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
