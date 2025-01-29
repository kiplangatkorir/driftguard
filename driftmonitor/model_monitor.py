import os
import numpy as np
class ModelMonitor:
    def __init__(self, model, metric="accuracy"):
        """
        Initializes the model monitor.
        :param model: The machine learning model to monitor.
        :param metric: The metric used to monitor model performance (default is "accuracy").
        """
        self.model = model
        self.metric = metric
        self.performance_history = []

    def track_performance(self, data, labels):
        """
        Tracks the performance of the model on new data.
        :param data: The input data.
        :param labels: The true labels.
        :return: A dictionary with the metric value.
        """
        metric_value = self._compute_metric(data, labels)
        self.performance_history.append(metric_value)
        
        return {self.metric: metric_value}

    def _compute_metric(self, data, labels):
        """
        Computes the chosen metric for model performance.
        :param data: The input data.
        :param labels: The true labels.
        :return: The computed metric value (default is accuracy).
        """
        predictions = self.model.predict(data)
        
        correct_predictions = np.sum(predictions == labels)
        accuracy = correct_predictions / len(labels)
        
        return accuracy
