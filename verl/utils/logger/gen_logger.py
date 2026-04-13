

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb


if is_package_available("swanlab"):
    import swanlab


@dataclass
class GenerationLogger(ABC):
    config: dict[str, Any]

    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for inp, out, lab, score in samples:
            print(f"[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n")


@dataclass
class FileGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        with open(os.path.join(self.config["trainer"]["save_checkpoint_path"], "generations.log"), "a") as f:
            for inp, out, lab, score in samples:
                f.write(f"[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n\n")


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"label_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))],
            [],
        )

        if not hasattr(self, "validation_table"):
            self.validation_table = wandb.Table(columns=columns)

        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        row_data = [step]
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = "\n\n---\n\n".join(
                (f"input: {sample[0]}", f"output: {sample[1]}", f"label: {sample[2]}", f"score: {sample[3]}")
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({"val/generations": swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "file": FileGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str], config: Optional[dict[str, Any]] = None):
        self.loggers: List[GenerationLogger] = []

        for logger in loggers:
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger](config))

    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for logger in self.loggers:
            logger.log(samples, step)
