from avalanche.evaluation import PluginMetric
# Import ajustado de `from F1Score import F1Score` (absoluto) para relativo,
# única mudança necessária para o arquivo original funcionar dentro do
# pacote `har_cl_module` (que usa imports relativos). Nenhuma lógica foi
# alterada.
from .F1Score import F1Score
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

class MinibatchF1Score(PluginMetric[float]):
    """
    Métrica de F1-Score calculada a cada iteração (minibatch).
    """
    def __init__(self):
        super().__init__()
        # Instancia sua métrica base (ex: F1Score do Avalanche ou a sua custom)
        self._f1_score = F1Score()

    def reset(self) -> None:
        self._f1_score.reset()

    def result(self) -> float:
        return self._f1_score.result()

    def after_training_iteration(self, strategy: 'TemplateStrategy'):
        # strategy.mb_y: labels reais do minibatch
        # strategy.mb_output: predições (logits) do modelo
        self._f1_score.update(strategy.mb_output, strategy.mb_y)
        return self._package_result(strategy)

    def _package_result(self, strategy: 'TemplateStrategy'):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=True)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "F1_Minibatch"