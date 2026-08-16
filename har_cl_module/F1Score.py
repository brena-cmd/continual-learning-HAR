import torch
from torch import Tensor
from typing import List, Optional, Union, Dict
from collections import defaultdict
from avalanche.evaluation import Metric, GenericPluginMetric

################################################################################
# STANDALONE F1-SCORE METRIC
################################################################################

class F1Score(Metric[float]):
    """Métrica F1-Score Standalone.
    Calcula o F1-Score acumulando os contadores de TP, FP e FN para evitar 
    erros matemáticos de média simples entre batches.
    """

    def __init__(self, average='macro'):
        super().__init__()
        self.average = average
        self._tp = defaultdict(float)
        self._fp = defaultdict(float)
        self._fn = defaultdict(float)
        self._support = defaultdict(float)

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor) -> None:
        # Mover para CPU e converter para tensor
        true_y = torch.as_tensor(true_y).cpu()
        predicted_y = torch.as_tensor(predicted_y).cpu()

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Transformar logits/probabilidades em labels se necessário
        if predicted_y.dim() > 1 and predicted_y.shape[1] > 1:
            predicted_y = predicted_y.argmax(dim=1)
        if true_y.dim() > 1 and true_y.shape[1] > 1:
            true_y = true_y.argmax(dim=1)

        # Garantir que são 1D
        predicted_y = predicted_y.flatten()
        true_y = true_y.flatten()

        # Sincronizar classes presentes
        classes = torch.unique(torch.cat((predicted_y, true_y)))
        for cls in classes:
            c = cls.item()
            self._tp[c] += torch.sum((predicted_y == c) & (true_y == c)).item()
            self._fp[c] += torch.sum((predicted_y == c) & (true_y != c)).item()
            self._fn[c] += torch.sum((true_y == c) & (predicted_y != c)).item()
            self._support[c] += torch.sum(true_y == c).item()

    def result(self) -> float:
        all_classes = set(self._tp.keys())
        if not all_classes:
            return 0.0

        f1_scores = {}
        total_support = sum(self._support.values())
        
        for c in all_classes:
            tp, fp, fn = self._tp[c], self._fp[c], self._fn[c]
            # Adicionar epsilon para evitar divisão por zero
            precision = tp / max(tp + fp, 1e-10)
            recall = tp / max(tp + fn, 1e-10)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            f1_scores[c] = f1

        if self.average == 'macro':
            return sum(f1_scores.values()) / len(f1_scores)
        
        if self.average == 'weighted':
            if total_support == 0: 
                return 0.0
            return sum(f1_scores[c] * (self._support[c] / total_support) for c in all_classes)

        return 0.0

    def reset(self) -> None:
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()
        self._support.clear()

################################################################################
# TASK-AWARE F1-SCORE METRIC
################################################################################

class TaskAwareF1(Metric[Dict[int, float]]):
    """Métrica F1-Score que mantém contadores separados por Task ID."""

    def __init__(self, average='macro'):
        super().__init__()
        self.average = average
        self._f1_metrics = defaultdict(lambda: F1Score(average=self.average))

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor, task_labels: Union[int, Tensor]) -> None:
        if isinstance(task_labels, int):
            # Task único para todo o batch
            self._f1_metrics[task_labels].update(predicted_y, true_y)
        elif isinstance(task_labels, Tensor):
            if len(task_labels) != len(true_y):
                raise ValueError("Size mismatch for true_y and task_labels")
            
            # Agrupar por task para fazer updates em batch (mais eficiente)
            task_labels_cpu = task_labels.cpu()
            unique_tasks = torch.unique(task_labels_cpu)
            
            for task_id in unique_tasks:
                task_id_val = task_id.item()
                mask = task_labels_cpu == task_id
                
                # Extrair predições e labels desta task
                if predicted_y.dim() > 1:
                    pred_task = predicted_y[mask]
                else:
                    pred_task = predicted_y[mask]
                    
                true_task = true_y[mask]
                
                self._f1_metrics[task_id_val].update(pred_task, true_task)

    def result(self, task_label: Optional[int] = None) -> Dict[int, float]:
        if task_label is None:
            return {k: v.result() for k, v in self._f1_metrics.items()}
        return {task_label: self._f1_metrics[task_label].result()}

    def reset(self, task_label: Optional[int] = None) -> None:
        if task_label is None:
            self._f1_metrics.clear()
        else:
            self._f1_metrics[task_label].reset()

################################################################################
# PLUGIN METRICS
################################################################################

class F1PluginMetric(GenericPluginMetric[float, F1Score]):
    """Classe base para Plugins de F1-Score."""
    def __init__(self, reset_at, emit_at, mode, average='macro'):
        super().__init__(F1Score(average=average), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None: 
        self._metric.reset()
        
    def result(self) -> float: 
        return self._metric.result()
        
    def update(self, strategy): 
        self._metric.update(strategy.mb_output, strategy.mb_y)

class F1PerTaskPluginMetric(GenericPluginMetric[Dict[int, float], TaskAwareF1]):
    """Classe base para Plugins de F1-Score sensíveis a Tarefas."""
    def __init__(self, reset_at, emit_at, mode, average='macro'):
        super().__init__(TaskAwareF1(average=average), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None: 
        self._metric.reset()
        
    def result(self) -> Dict[int, float]: 
        return self._metric.result()
        
    def update(self, strategy): 
        self._metric.update(strategy.mb_output, strategy.mb_y, strategy.mb_task_id)

# Implementações Específicas
class EpochF1(F1PluginMetric):
    def __init__(self, average='macro'):
        super().__init__(reset_at="epoch", emit_at="epoch", mode="train", average=average)
    def __str__(self): 
        return f"F1_Epoch_{self._metric.average}"

class ExperienceF1(F1PluginMetric):
    def __init__(self, average='macro'):
        super().__init__(reset_at="experience", emit_at="experience", mode="eval", average=average)
    def __str__(self): 
        return f"F1_Exp_{self._metric.average}"

class ExperienceF1PerTask(F1PerTaskPluginMetric):
    def __init__(self, average='macro'):
        super().__init__(reset_at="experience", emit_at="experience", mode="eval", average=average)
    def __str__(self): 
        return f"F1_Exp_PerTask_{self._metric.average}"

class StreamF1(F1PluginMetric):
    def __init__(self, average='macro'):
        super().__init__(reset_at="stream", emit_at="stream", mode="eval", average=average)
    def __str__(self): 
        return f"F1_Stream_{self._metric.average}"

################################################################################
# HELPER METHOD
################################################################################

def f1_metrics(
    *,
    epoch=False,
    experience=False,
    stream=False,
    per_task=False,
    average='macro'
) -> List[Union[F1PluginMetric, F1PerTaskPluginMetric]]:
    """Helper para obter métricas de F1-Score.
    
    Args:
        epoch: Se True, inclui métrica por época (treino)
        experience: Se True, inclui métrica por experiência (avaliação)
        stream: Se True, inclui métrica por stream (avaliação)
        per_task: Se True, calcula F1 separado por task (apenas com experience=True)
        average: 'macro' ou 'weighted'
    
    Returns:
        Lista de métricas configuradas
    """
    metrics = []
    if epoch:
        metrics.append(EpochF1(average=average))
    if experience:
        if per_task:
            metrics.append(ExperienceF1PerTask(average=average))
        else:
            metrics.append(ExperienceF1(average=average))
    if stream:
        metrics.append(StreamF1(average=average))
    return metrics
# import torch
# from torch import Tensor
# from typing import List, Optional, Union, Dict
# from collections import defaultdict
# from avalanche.evaluation import Metric, GenericPluginMetric

# ################################################################################
# # STANDALONE F1-SCORE METRIC
# ################################################################################

# class F1Score(Metric[float]):
#     """Métrica F1-Score Standalone.
#     Calcula o F1-Score acumulando os contadores de TP, FP e FN para evitar 
#     erros matemáticos de média simples entre batches.
#     """

#     def __init__(self, average='macro'):
#         super().__init__()
#         self.average = average
#         self._tp = defaultdict(float)
#         self._fp = defaultdict(float)
#         self._fn = defaultdict(float)
#         self._support = defaultdict(float)

#     @torch.no_grad()
#     def update(self, predicted_y: Tensor, true_y: Tensor) -> None:
#         true_y = torch.as_tensor(true_y)
#         predicted_y = torch.as_tensor(predicted_y)

#         if len(true_y) != len(predicted_y):
#             raise ValueError("Size mismatch for true_y and predicted_y tensors")

#         # Transformar logits/probabilidades em labels se necessário
#         if len(predicted_y.shape) > 1:
#             predicted_y = torch.max(predicted_y, 1)[1]
#         if len(true_y.shape) > 1:
#             true_y = torch.max(true_y, 1)[1]

#         # Sincronizar classes presentes
#         classes = torch.unique(torch.cat((predicted_y, true_y)))
#         for cls in classes:
#             c = cls.item()
#             self._tp[c] += torch.sum((predicted_y == c) & (true_y == c)).item()
#             self._fp[c] += torch.sum((predicted_y == c) & (true_y != c)).item()
#             self._fn[c] += torch.sum((true_y == c) & (predicted_y != c)).item()
#             self._support[c] += torch.sum(true_y == c).item()

#     def result(self) -> float:
#         all_classes = set(self._tp.keys())
#         if not all_classes:
#             return 0.0

#         f1_scores = {}
#         total_support = sum(self._support.values())
        
#         for c in all_classes:
#             tp, fp, fn = self._tp[c], self._fp[c], self._fn[c]
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#             f1_scores[c] = f1

#         if self.average == 'macro':
#             return sum(f1_scores.values()) / len(f1_scores)
        
#         if self.average == 'weighted':
#             if total_support == 0: return 0.0
#             return sum(f1_scores[c] * (self._support[c] / total_support) for c in all_classes)

#         return 0.0

#     def reset(self) -> None:
#         self._tp.clear()
#         self._fp.clear()
#         self._fn.clear()
#         self._support.clear()

# ################################################################################
# # TASK-AWARE F1-SCORE METRIC
# ################################################################################

# class TaskAwareF1(Metric[Dict[int, float]]):
#     """Métrica F1-Score que mantém contadores separados por Task ID."""

#     def __init__(self, average='macro'):
#         super().__init__()
#         self.average = average
#         self._f1_metrics = defaultdict(lambda: F1Score(average=self.average))

#     @torch.no_grad()
#     def update(self, predicted_y: Tensor, true_y: Tensor, task_labels: Union[int, Tensor]) -> None:
#         if isinstance(task_labels, int):
#             self._f1_metrics[task_labels].update(predicted_y, true_y)
#         elif isinstance(task_labels, Tensor):
#             if len(task_labels) != len(true_y):
#                 raise ValueError("Size mismatch for true_y and task_labels")
#             for pred, true, t in zip(predicted_y, true_y, task_labels):
#                 t_id = t.item() if isinstance(t, Tensor) else t
#                 self._f1_metrics[t_id].update(pred.unsqueeze(0), true.unsqueeze(0))

#     def result(self, task_label: Optional[int] = None) -> Dict[int, float]:
#         if task_label is None:
#             return {k: v.result() for k, v in self._f1_metrics.items()}
#         return {task_label: self._f1_metrics[task_label].result()}

#     def reset(self, task_label: Optional[int] = None) -> None:
#         if task_label is None:
#             self._f1_metrics.clear()
#         else:
#             self._f1_metrics[task_label].reset()

# ################################################################################
# # PLUGIN METRICS
# ################################################################################

# class F1PluginMetric(GenericPluginMetric[float, F1Score]):
#     """Classe base para Plugins de F1-Score."""
#     def __init__(self, reset_at, emit_at, mode, average='macro'):
#         super().__init__(F1Score(average=average), reset_at=reset_at, emit_at=emit_at, mode=mode)

#     def reset(self) -> None: self._metric.reset()
#     def result(self) -> float: return self._metric.result()
#     def update(self, strategy): self._metric.update(strategy.mb_output, strategy.mb_y)

# class F1PerTaskPluginMetric(GenericPluginMetric[Dict[int, float], TaskAwareF1]):
#     """Classe base para Plugins de F1-Score sensíveis a Tarefas."""
#     def __init__(self, reset_at, emit_at, mode, average='macro'):
#         super().__init__(TaskAwareF1(average=average), reset_at=reset_at, emit_at=emit_at, mode=mode)

#     def reset(self) -> None: self._metric.reset()
#     def result(self) -> Dict[int, float]: return self._metric.result()
#     def update(self, strategy): self._metric.update(strategy.mb_output, strategy.mb_y, strategy.mb_task_id)

# # Implementações Específicas
# class EpochF1(F1PluginMetric):
#     def __init__(self, average='macro'):
#         super().__init__(reset_at="epoch", emit_at="epoch", mode="train", average=average)
#     def __str__(self): return f"F1_Epoch_{self._metric.average}"

# class ExperienceF1(F1PluginMetric):
#     def __init__(self, average='macro'):
#         super().__init__(reset_at="experience", emit_at="experience", mode="eval", average=average)
#     def __str__(self): return f"F1_Exp_{self._metric.average}"

# class ExperienceF1PerTask(F1PerTaskPluginMetric):
#     def __init__(self, average='macro'):
#         super().__init__(reset_at="experience", emit_at="experience", mode="eval", average=average)
#     def __str__(self): return f"F1_Exp_PerTask_{self._metric.average}"

# class StreamF1(F1PluginMetric):
#     def __init__(self, average='macro'):
#         super().__init__(reset_at="stream", emit_at="stream", mode="eval", average=average)
#     def __str__(self): return f"F1_Stream_{self._metric.average}"

# ################################################################################
# # HELPER METHOD
# ################################################################################

# def f1_metrics(
#     *,
#     epoch=False,
#     experience=False,
#     stream=False,
#     per_task=False,
#     average='macro'
# ) -> List[Union[F1PluginMetric, F1PerTaskPluginMetric]]:
#     """Helper para obter métricas de F1-Score."""
#     metrics = []
#     if epoch:
#         metrics.append(EpochF1(average=average))
#     if experience:
#         if per_task:
#             metrics.append(ExperienceF1PerTask(average=average))
#         else:
#             metrics.append(ExperienceF1(average=average))
#     if stream:
#         metrics.append(StreamF1(average=average))
#     return metrics