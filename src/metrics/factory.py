from src.metrics.classification import AccuracyMetric, RocAucMetric, LogLossMetric

class MetricFactory:
    _registry = {
        "accuracy": AccuracyMetric,
        "roc_auc": RocAucMetric,
        "log_loss": LogLossMetric
    }

    @staticmethod
    def get_metrics(metric_names: list):
        """Возвращает список инициализированных классов метрик"""
        metrics = []
        for name in metric_names:
            if name not in MetricFactory._registry:
                print(f"Warning: Metric '{name}' not found in registry.")
                continue
            metrics.append(MetricFactory._registry[name]())
        return metrics
