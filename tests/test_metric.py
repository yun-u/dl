import numpy as np
import sklearn

from dl.logger.metric import is_metric_fn


def accuracy_score(labels: np.ndarray, logits: np.ndarray):
    return sklearn.metrics.accuracy_score(labels, logits.argmax(axis=1))


def confusion_matrix(labels: np.ndarray, logits: np.ndarray):
    return sklearn.metrics.confusion_matrix(labels, logits.argmax(axis=1))


def sample_metric(labels: np.ndarray, logits: np.ndarray, *args, **kwargs):
    pass


def test_is_metric_fn():
    assert is_metric_fn(accuracy_score)
    assert is_metric_fn(confusion_matrix)
    assert is_metric_fn(sample_metric)
