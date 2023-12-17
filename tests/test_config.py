import json
import logging

from dl.trainer import Config


def test_config():
    json_data = """
    {
        "seed": 42,
        "device": 0,
        "output_dir": null,
        "max_iters": 256,
        "eval_iters": 100,
        "eval_interval": 10,
        "train_batch_size": 128,
        "eval_batch_size": 128,
        "optimizer": ["torch.optim.SGD", {"lr": 3e-4}],
        "weight_decay": 0.0,
        "lr_scheduler": ["dl.lr_scheduler.WarmupCosineLR", {"warmup_iters": 10, "lr_decay_iters": 256, "min_lr": 1e-5}],
        "metrics": [["sklearn.metrics.accuracy_score", {}]],
        "dataloader": {
            "num_workers": 8,
            "pin_memory": true

        },
        "mlflow": {
            "expermient": "tests"
        }
    }
    """

    obj = json.loads(json_data)
    cfg = Config.from_dict(obj)
    logging.info(cfg.model_dump_json(indent=2))
