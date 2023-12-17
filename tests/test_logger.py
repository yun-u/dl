import logging

import numpy as np
import sklearn

from dl.logger import Log, Logger


def test_matched_groups():
    from dl.logger.utils import matched_groups

    assert list(matched_groups(["x/y_true", "x/y_pred"], r"(\w+)/(\w+)")) == [
        ("x", "y_true"),
        ("x", "y_pred"),
    ]


def test_log():
    log = Log()
    log.collect(
        0,
        labels=[0, 1],
        logits=[[0.9, 0.1], [0.2, 0.8]],
        k1={"k1": "v1"},
        k2=[100],
        k3=2,
    )
    log.collect(
        1,
        labels=[0],
        logits=[[0.7, 0.3]],
        k1={"k2": "v2"},
        k2=[101],
        k3=5,
    )

    logging.info(log.last())
    logging.info(log.merge())


def accuracy_score(labels: np.ndarray, logits: np.ndarray, **kwargs):
    return sklearn.metrics.accuracy_score(labels, logits.argmax(axis=1))


def confusion_matrix(labels: np.ndarray, logits: np.ndarray, **kwargs):
    return sklearn.metrics.confusion_matrix(labels, logits.argmax(axis=1))


def test_prepare_log():
    logger = Logger(10, [accuracy_score, confusion_matrix], [])

    logger.collect(0, "train", labels=[0, 1], logits=[[0.9, 0.1], [0.2, 0.8]])
    logger.collect(1, "train", labels=[0], logits=[[0.7, 0.3]])

    logging.info(logger.prepare_log(prefix="train", log_type="batch", lr=3e-4))

    logger.collect(0, "val", labels=[0], logits=[[0.5, 0.5]])
    logger.collect(1, "val", labels=[1], logits=[[0.5, 0.5]])

    logging.info(logger.prepare_log(prefix="val", log_type="batch"))

    logger.collect(0, "val", labels=[1], logits=[[1.0, 0.0]])
    logger.collect(1, "val", labels=[1], logits=[[1.0, 0.0]])
    logger.collect(0, "x", labels=[1], logits=[[1.0, 0.0]])

    logging.info(logger.log(prefix=("train", "val", "x"), log_type="epoch", epoch=100))
