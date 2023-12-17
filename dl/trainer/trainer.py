import functools
import os
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import mlflow
import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler
from typing_extensions import NotRequired

from dl.logger import Logger
from dl.logger.callback import Callback, Mlflow
from dl.logging_setup import LOG_FORMAT, get_logger
from dl.lr_scheduler import LRScheduler
from dl.utils.git import add_and_commit_changes, is_git_repository
from dl.utils.pytorch import worker_init_function

from .config import Config

__all__ = [
    "LossAndLogits",
    "Trainer",
]

logger = get_logger()


class LossAndLogits(TypedDict):
    loss: NotRequired[Optional[torch.Tensor]]  # loss value
    logits: torch.Tensor  # logits from the model


class Trainer:
    """The Trainer.

    Args:
        model (`nn.Module`): The neural network model to train.
        config (`Config`): Configuration settings for training.
        train_dataset (`Dataset`): The training dataset.
        eval_dataset (`Dataset`): The evaluation dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[Config, str, os.PathLike],
        train_dataset: Dataset,
        eval_dataset: Union[Dataset, Dict[str, Dataset]],
    ) -> None:
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, (str, os.PathLike)):
            self.config = Config.from_file(config)

        self.run = self.config.mlflow.create_run()
        self.setup_logging()

        self.train_dataset = train_dataset

        if isinstance(eval_dataset, Dataset):
            self.eval_dataset = {"val": eval_dataset}
        elif isinstance(eval_dataset, dict):
            self.eval_dataset = eval_dataset

        self.logger = Logger(
            self.config.max_iters, self.config.get_metrics(), [Mlflow()]
        )
        self.device = torch.device(self.config.device)

        self.model = self.create_model(model)
        self.to(self.device)

        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler(self.optimizer)

    def setup_logging(self) -> None:
        logger.add(
            sink=Path(self.output_dir, "logs", "{time:YYYY-MM-DDTHH:00:00Z}.log"),
            format=LOG_FORMAT,
            colorize=False,
            backtrace=True,
            rotation=timedelta(hours=1),
        )

    @functools.cached_property
    def output_dir(self) -> Path:
        """Get the output directory for saving models and logs.

        Returns:
            `Path`: The absolute path to the output directory.
        """
        if self.config.output_dir is not None:
            return Path(self.config.output_dir)

        output_dir = self.config.mlflow.artifact_dir(self.run).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    ################################################################################
    # Save & Load
    ################################################################################

    @functools.cached_property
    def model_path(self) -> Path:
        path = self.output_dir / "model" / "pytorch_model.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def optimizer_path(self) -> Path:
        path = self.output_dir / "model" / "optimizer.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def config_path(self) -> Path:
        path = self.output_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_model(self) -> None:
        """Save the model to a file."""
        save_model(self.model, self.model_path)

    def save_optimizer(self) -> None:
        """Save the optimizer's state to a file."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)

    def save_config(self) -> None:
        """Save the training configuration to a JSON file."""
        self.config_path.write_text(self.config.model_dump_json(indent=2))

    def save(self) -> None:
        """Save the model, optimizer, and configuration."""
        self.save_model()
        self.save_optimizer()
        self.save_config()

    def load_model(self, model: nn.Module) -> nn.Module:
        """Load the model from a file."""
        load_model(model, self.model_path)
        return model

    def load_optimizer(self) -> Dict[str, Any]:
        """Load the optimizer's state from a file."""
        return torch.load(self.optimizer_path)

    def load_config(self) -> Config:
        """Load the training configuration from a file."""
        return Config.from_file(self.config_path)

    ################################################################################
    # Create
    ################################################################################

    def create_model(self, model: nn.Module) -> nn.Module:
        """Create the model.

        Args:
            model (`nn.Module`): The model.

        Returns:
            `nn.Module`: The created model.
        """
        if self.config.init_mode == "resume":
            return self.load_model(model)
        return model

    def to(self, device: torch.device):
        """Move the model to the specified device.

        Args:
            device (`torch.device`): The target device (e.g., 'cuda' or 'cpu').
        """
        logger.info(f"Moving the model to '{device}'.")
        self.model.to(device)

    def create_optimizer(self) -> Optimizer:
        """Create the optimizer for training.

        Returns:
            `Optimizer`: The optimizer.
        """
        optimizer = self.config.get_optimizer(
            params=self.get_param_groups(
                self.model, weight_decay=self.config.weight_decay
            ),
        )

        # TODO: merge resume at once
        if self.config.init_mode == "resume":
            optimizer.load_state_dict(self.load_optimizer())

        return optimizer

    def create_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Create a learning rate scheduler.

        Args:
            optimizer (`Optimizer`): The optimizer.

        Returns:
            `LRScheduler`: The learning rate scheduler.
        """
        last_epoch = (
            -1 if self.config.init_mode == "scratch" else self.config.num_iter - 1
        )

        return self.config.get_lr_scheduler(optimizer, last_epoch)

    def get_param_groups(
        self,
        model: nn.Module,
        weight_decay: float,
        whitelist: Tuple[nn.Module, ...] = (torch.nn.Linear, torch.nn.Conv2d),
        blacklist: Tuple[nn.Module, ...] = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.modules.batchnorm._BatchNorm,
        ),
    ) -> List[Dict[str, Any]]:
        """Get parameter groups for the optimizer.

        Args:
            model (`nn.Module`): The model.
            weight_decay (`float`): Weight decay for regularization.
            whitelist (`Tuple[nn.Module, ...]`, optional): Modules to include in weight decay.`.
            blacklist (`Tuple[nn.Module, ...]`, optional): Modules to exclude from weight decay.`.

        Returns:
            `List[Dict[str, Any]]`: List of parameter groups for the optimizer.
        """
        # refer to https://github.com/karpathy/nanoGPT/blob/196160b849eaed4465310c082db2c9dc7bc11ba9/model.py#L270
        decay, no_decay = set(), set()

        for module_name, module in model.named_modules():
            for param_name, _ in module.named_parameters():
                if module_name == "":
                    full_param_name = param_name
                else:
                    full_param_name = f"{module_name}.{param_name}"

                if param_name.endswith("bias"):
                    no_decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist):
                    decay.add(full_param_name)
                    if full_param_name in no_decay:
                        no_decay.remove(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist):
                    no_decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)

        param_dict = {
            param_name: param for param_name, param in model.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert (
            len(inter_params) == 0
        ), f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"Parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        param_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return param_groups

    ################################################################################
    # DataLoader
    ################################################################################

    def get_train_dataloader(self) -> DataLoader:
        """Get the training data loader.

        Returns:
            `DataLoader`: The training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_function,
        )

    def get_eval_dataloader(
        self, eval_dataset: Dataset, is_in_train: bool = False
    ) -> DataLoader:
        """Get the evaluation data loader.

        Args:
            eval_dataset (`Dataset`): The evaluation dataset.
            is_in_train (`bool`): Whether evaluation is performed during training. Defaults to False.

        Returns:
            `DataLoader`: The evaluation data loader.
        """
        return DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            sampler=self._get_eval_sampler(eval_dataset, is_in_train),
            drop_last=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_function,
        )

    def _get_eval_sampler(
        self, eval_dataset: Dataset, is_in_train: bool
    ) -> Optional[Sampler]:
        """Get the evaluation sampler based on the current training configuration.

        Args:
            eval_dataset (`Dataset`): The evaluation dataset.
            is_in_train (`bool`): Whether evaluation is performed during training.

        Returns:
            `Optional[Sampler]`: The evaluation sampler or None if not applicable.
        """
        if is_in_train and self.config.eval_iters > 0:
            num_samples = self.config.eval_iters * self.config.eval_batch_size

            if num_samples > len(eval_dataset):
                raise ValueError(
                    "The number of samples exceeds the size of evaluation datset."
                )

            return RandomSampler(eval_dataset, num_samples=num_samples)
        return None

    def prepare_inputs(self, inputs: Any) -> Any:
        """Prepare inputs for the model by moving them to the device.

        Args:
            inputs (`Any`): Input data.

        Returns:
            `Any`: Prepared input data.
        """
        if isinstance(inputs, Mapping):
            return type(inputs)({k: self.prepare_inputs(v) for k, v in inputs.items()})
        elif isinstance(inputs, Sequence):
            return type(inputs)(self.prepare_inputs(v) for v in inputs)
        elif isinstance(inputs, torch.Tensor):
            # TODO: non_blocking=True
            return inputs.to(device=self.device, non_blocking=True)
        return inputs

    ################################################################################
    # Train
    ################################################################################

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> LossAndLogits:
        """Compute the loss and logits.

        Args:
            model (`nn.Module`): The model.
            inputs (`Dict[str, torch.Tensor]`): Input data.

        Returns:
            `LossAndLogits`: Loss and logits. If there is no 'labels' key in inputs, the loss is not returned.
        """
        raise NotImplementedError

    def train(self, callbacks: Sequence[Callback] = ()):
        """Train the model."""
        if not is_git_repository(Path().cwd()):
            raise ValueError(
                "The current directory is not a Git repository. Please make sure you are in a valid Git repository."
            )

        add_and_commit_changes(self.run.info.run_name)
        logger.info(f"Create a new commit '{self.run.info.run_name}'")

        with mlflow.start_run(run_id=self.run.info.run_id):
            mlflow.log_params(self.config.model_dump())

            logger.info(f"Set working directory to '{self.output_dir}'")

            if self.config.init_mode == "resume":
                logger.info(f"Resuming training from '{self.config.output_dir}'.")

            self.logger.add_callbacks(*callbacks)

            while True:
                if self._is_training_completed(self.config.num_iter):
                    break

                self.training_loop()

                if self.eval_dataset is not None or self.config.eval_iters == 0:
                    for prefix, eval_dataset in self.eval_dataset.items():
                        self.evaluation_loop(
                            self.get_eval_dataloader(eval_dataset, is_in_train=False),
                            prefix=prefix,
                        )

                eval_prefixes = tuple(sorted(self.eval_dataset.keys()))
                self.logger.log(
                    prefix=("train",) + eval_prefixes,
                    log_type="epoch",
                    epoch=self.config.num_epoch,
                )

                self.config.num_epoch += 1

                self.save()

    def training_loop(self) -> None:
        """The main training loop."""
        if self.optimizer is None:
            raise ValueError("The optimizer is not set up.")

        for batch in self.get_train_dataloader():
            if self._is_training_completed(self.config.num_iter):
                break

            lr = self.optimizer.param_groups[0]["lr"]

            loss_and_logits = self.training_step(self.model, batch)
            loss, logits = loss_and_logits["loss"], loss_and_logits["logits"]

            self.logger.collect(
                index=self.config.num_iter,
                prefix="train",
                weight=len(logits),
                **{
                    "loss": loss,
                    "logits": logits,
                    "labels": batch.get("labels"),
                    **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                },
            )

            self.logger.log(prefix="train", log_type="batch", lr=lr)

            if (
                self.eval_dataset is not None
                and self.config.num_iter % self.config.eval_interval == 0
            ):
                for prefix, eval_dataset in self.eval_dataset.items():
                    self.evaluation_loop(
                        self.get_eval_dataloader(eval_dataset, is_in_train=True),
                        prefix=prefix,
                    )

                eval_prefixes = tuple(sorted(self.eval_dataset.keys()))
                self.logger.log(
                    prefix=eval_prefixes,
                    log_type="batch",
                    iter=self.config.num_iter,
                )

            self.config.num_iter += 1

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> LossAndLogits:
        """Perform a single training step.

        Args:
            model (`nn.Module`): The model to train.
            inputs (`Dict[str, torch.Tensor]`): Input data.

        Returns:
            `LossAndLogits`: Loss and logits from the model.
        """
        model.train()

        inputs = self.prepare_inputs(inputs)

        loss_and_logits = self.compute_loss(model, inputs)
        loss, logits = loss_and_logits["loss"], loss_and_logits["logits"]

        loss.backward()

        self.optimizer.step()
        self.lr_scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)  # TODO: model.zero_grad ?

        return {"loss": loss.detach(), "logits": logits.detach()}

    def _is_training_completed(self, num_iter: int) -> bool:
        """Checks if the training is completed based on the given num_iter.

        Args:
            num_iter (`int`): The current iteration. If it is less than 0, the training will continue.

        Returns:
            `bool`: True if training is completed, False otherwise.
        """
        if num_iter >= self.config.max_iters >= 0:
            return True

        return False

    ################################################################################
    # Evaluate & Predict
    ################################################################################

    def evaluate(
        self,
        eval_dataset: Dataset,
        prefix: Literal["val", "test"] = "val",
    ) -> Dict[str, Any]:
        """Evaluate the model.

        Args:
            eval_dataset (`Dataset`): The evaluation dataset.
            prefix (`Literal["val", "test"]`): Prefix for logging. Defaults to "val".
        Returns:
            `Dict[str, Any]`: Evaluation results.
        """
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.evaluation_loop(dataloader, prefix)
        return self.logger.prepare_log(
            prefix=prefix, log_type="epoch", exclude=None, epoch=0
        )

    def evaluation_loop(
        self, dataloader: DataLoader, prefix: Union[str, Literal["val", "test"]] = "val"
    ) -> None:
        """The main evaluation loop.

        Args:
            dataloader (`DataLoader`): The data loader for evaluation.
            prefix (`Union[str, Literal["val", "test"]]`): Prefix for logging. Defaults to "val".
        """
        for i, batch in enumerate(dataloader):
            loss_and_logits = self.evaluation_step(self.model, batch)
            loss, logits = loss_and_logits.get("loss"), loss_and_logits.get("logits")

            # if 'batch' does not have 'labels', logger will not collect 'labels'.
            self.logger.collect(
                index=i,
                prefix=prefix,
                weight=len(logits),
                **{
                    "loss": loss,
                    "logits": logits,
                    "labels": batch.get("labels"),
                    **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                },
            )

    @torch.no_grad()
    def evaluation_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> LossAndLogits:
        """Perform a single evaluation step.

        Args:
            model (`nn.Module`): The model to evaluate.
            inputs (`Dict[str, torch.Tensor]`): Input data.

        Returns:
            `LossAndLogits`: Loss and logits from the model.
        """
        model.eval()

        inputs = self.prepare_inputs(inputs)

        return self.compute_loss(model, inputs)

    ################################################################################
    # Debugging
    ################################################################################

    def get_fixed_train_dataloader(self, num_samples: Union[int, float]) -> DataLoader:
        """Get a fixed-size training data loader for debugging.

        Args:
            num_samples (`Union[int, float]`): Number of samples or fraction of the dataset.

        Returns:
            `DataLoader`: The fixed-size training data loader.
        """
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        batch_sampler = BatchSampler(
            RandomSampler(
                self.train_dataset,
                num_samples=self._get_num_samples(num_samples),
                generator=generator,
            ),
            batch_size=self.config.train_batch_size,
            drop_last=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_function,
        )

    def overfit(self, num_samples: Union[int, float]):
        """Overfit the model on a small dataset for debugging.

        Args:
            num_samples (`Union[int, float]`): Number of samples or fraction of the dataset.
        """
        assert self.eval_dataset is None or self.config.eval_iters == 0
        self.get_train_dataloader = functools.partial(
            self.get_fixed_train_dataloader, num_samples
        )
        logger.info(f"Overfitting {self._get_num_samples(num_samples)} samples.")
        self.train()

    def _get_num_samples(self, num_samples: Union[int, float]) -> int:
        """Calculate the number of samples to use for debugging.

        Args:
            num_samples (`Union[int, float]`): Number of samples or fraction of the dataset.

        Returns:
            `int`: Number of samples.
        """
        if num_samples < 1.0:
            return int(len(self.train_dataset) * num_samples)
        return num_samples
