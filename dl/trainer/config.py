from __future__ import annotations

import functools
import json
import urllib.parse
import uuid
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import mlflow
import pydantic
from mlflow.entities import Run
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import Annotated

from dl.logger.metric import Metric
from dl.logging_setup import get_logger
from dl.utils.generic import get_object

__all__ = [
    "Config",
    "DataLoaderConfig",
]

logger = get_logger()


ModuleNameAndParams = Tuple[str, Dict[str, Any]]


def _convert_module(
    v: ModuleNameAndParams,
    globals: Optional[Dict] = None,
    locals: Optional[Dict] = None,
) -> Callable:
    target, kwargs = v
    return functools.partial(get_object(target, globals, locals), **kwargs)


class DataLoaderConfig(BaseModel):
    """Configuration for data loading in the training process.

    Args:
        num_workers (`int`): Number of worker processes for data loading.
        pin_memory (`bool`): Whether to use pinned memory for data transfer.
        train_transform (`ModuleNameAndParams`, optional): A module name and its parameters
            for augmentation transforms during training.
        eval_transform (`ModuleNameAndParams`, optional): A module name and its parameters
            for augmentation transforms during evaluation.
    """

    num_workers: int
    pin_memory: bool

    train_transform: Optional[ModuleNameAndParams] = None
    eval_transform: Optional[ModuleNameAndParams] = None

    def get_train_transform(self) -> Optional[Callable]:
        if self.train_transform:
            return _convert_module(self.train_transform)
        return None

    def get_eval_transform(self) -> Optional[Callable]:
        if self.eval_transform:
            return _convert_module(self.eval_transform)
        return None


class MlflowConfig(BaseModel):
    """Configuration for the Mlflow.

    Args:
        experiment (`str`, optional): The name of the experiment to set. If not given, it is set to the default experiment.
    """

    experiment: Optional[str] = None

    @property
    def experiment_id(self) -> str:
        experiment_name = self.experiment or "0"

        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        return experiment_id

    def create_run(self) -> Run:
        run = mlflow.MlflowClient().create_run(
            experiment_id=self.experiment_id, run_name=str(uuid.uuid4())
        )
        return run

    @staticmethod
    def artifact_dir(run: Run) -> Path:
        parsed_uri = urllib.parse.urlparse(run.info.artifact_uri)
        return Path(parsed_uri.path)


class Config(BaseModel):
    """Configuration for the training process.

    Args:
        seed (`int`): Seed value for random number generators.
        device (`Union[str, int]`): Device index or identifier (e.g., GPU) to be used.
        output_dir (`Union[str, PathLike, None]`): Output directory for checkpoints and logs. The path must be absolute.
            If `output_dir` is None, it will behave in scratch mode, and if a path is given, in resume mode.
        max_iters (`int`): Maximum number of iterations for training.
        eval_iters (`int`): Number of iterations for evaluations.
        eval_interval (`int`): Interval (in iterations) between evaluations.
        train_batch_size (`int`): Batch size for training.
        eval_batch_size (`int`): Batch size for evaluation.
        optimizer (`ModuleNameAndParams`): A module name and its parameters for the optimizer.
        weight_decay (`float`): Weight decay regularization factor for optimizer.
        lr_scheduler (`ModuleNameAndParams`, optional): Configuration for the learning rate scheduler.
        metrics (`List[ModuleNameAndParams]`): List of module name and its parameters for metrics to track during training.
        dataloader (`DataLoaderConfig`): Configuration for data loader.
    """

    seed: int
    device: Union[str, int]

    num_iter: Annotated[int, Field(ge=0)] = 0
    num_epoch: int = 0

    output_dir: Union[str, PathLike, None] = None

    max_iters: Annotated[int, Field(ge=1)]

    eval_iters: int
    eval_interval: Annotated[int, Field(ge=1)]

    train_batch_size: Annotated[int, Field(ge=1)]
    eval_batch_size: Annotated[int, Field(ge=1)]

    optimizer: ModuleNameAndParams
    weight_decay: float = 0.0

    lr_scheduler: Optional[ModuleNameAndParams] = None

    metrics: List[ModuleNameAndParams] = Field(default_factory=list)

    dataloader: DataLoaderConfig
    mlflow: MlflowConfig

    @field_validator("output_dir")
    @classmethod
    def must_be_absolute_path(cls, v: Union[str, PathLike, None]) -> Union[Path, None]:
        if v is None:
            return None
        return Path(v).absolute()

    @property
    def init_mode(self) -> Literal["scratch", "resume"]:
        if self.output_dir is None:
            return "scratch"
        return "resume"

    @model_validator(mode="after")
    def check_resume(self) -> Config:
        if self.init_mode == "resume" and not Path(self.output_dir).exists():
            raise ValueError("Output directory does not exist for resume mode.")

        if self.init_mode == "resume":
            logger.warning(
                "When resume, the provided `num_iter`, `num_epoch`, `optimizer` and `lr_scehduler` have no effect."
            )

        if not (
            self.init_mode == "scratch" and self.num_iter == 0 and self.num_epoch == 0
        ):
            raise ValueError("Invalid combination of parameters for the scratch mode.")
        return self

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        return pydantic.TypeAdapter(Config).validate_python(obj)

    @staticmethod
    def from_file(path: Union[str, PathLike]):
        obj = json.loads(Path(path).read_text())
        return Config.from_dict(obj)

    def get_optimizer(self, params: Iterable, **kwargs) -> Optimizer:
        return _convert_module(self.optimizer)(params=params, **kwargs)

    def get_lr_scheduler(
        self, optimizer: Optimizer, last_epoch: int = -1, **kwargs
    ) -> Optional[LRScheduler]:
        if self.lr_scheduler:
            return _convert_module(self.lr_scheduler)(
                optimizer=optimizer, last_epoch=last_epoch, **kwargs
            )
        return None

    def get_metrics(self) -> List[Metric]:
        return [_convert_module(metric) for metric in self.metrics]
