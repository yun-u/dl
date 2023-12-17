from typing import Any, Dict, Literal

import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset

from dl.logger.callback import BestMetric, History
from dl.logging_setup import get_logger
from dl.trainer import Config, DataLoaderConfig, LossAndLogits, Trainer

logger = get_logger()


class MyDataset(Dataset):
    def __init__(self, size: int, split: Literal["train", "test"] = "train") -> None:
        self.size = size
        self.split = split

    def __getitem__(self, index) -> Dict[str, Any]:
        index = torch.tensor([index], dtype=torch.float32)
        if self.split == "test":
            return {"inputs": index}
        return {"inputs": index, "labels": index * 10}

    def __len__(self) -> int:
        return self.size


class MyTrainer(Trainer):
    def compute_loss(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> LossAndLogits:
        logits = model(inputs["inputs"])

        if (labels := inputs.get("labels")) is None:
            return {"logits": logits}

        loss = torch.nn.functional.mse_loss(logits, labels)

        return {"loss": loss, "logits": logits}


max_iters = 10
config = Config(
    seed=42,
    device="cpu",
    max_iters=max_iters,
    eval_iters=-1,
    eval_interval=1,
    train_batch_size=25,
    eval_batch_size=10,
    optimizer=["torch.optim.SGD", {"lr": 3e-4}],
    weight_decay=0.0,
    lr_scheduler=[
        "dl.lr_scheduler.WarmupCosineLR",
        {"warmup_iters": 5, "lr_decay_iters": 15, "min_lr": 1e-5},
    ],
    dataloader=DataLoaderConfig(num_workers=4, pin_memory=False),
)

train_dataset = MyDataset(size=50)
eval_dataset = MyDataset(size=100)
test_dataset = MyDataset(size=10, split="test")


def accuracy_score(labels: np.ndarray, logits: np.ndarray):
    return sklearn.metrics.accuracy_score(labels, logits.argmax(axis=1))


def test_resume():
    model = torch.nn.Linear(1, 1)
    trainer = MyTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    trainer.logger.add_metric(accuracy_score)

    trainer.train(
        callbacks=[
            History("history.log"),
            BestMetric(
                metric_name="accuracy_score",
                mode="max",
                model=model,
                stage="train",
            ),
            BestMetric(
                metric_name="loss",
                mode="min",
                model=model,
                stage="val",
            ),
        ]
    )

    logger.info(list(model.parameters()))

    new_model = torch.nn.Linear(1, 1)
    new_config = config.model_copy()
    new_config.output_dir = str(trainer.output_dir)
    new_config.max_iters = 15
    new_trainer = MyTrainer(
        model=new_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=new_config,
    )
    new_trainer.train()
    logger.info(list(new_model.parameters()))
    # logger.info(trainer.evaluate())
    # logger.info(trainer.evaluate(test_dataset))


def test_overfit():
    eval_dataset = None

    model = torch.nn.Linear(1, 1)
    trainer = MyTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    trainer.overfit(8)
