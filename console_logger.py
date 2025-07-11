import logging

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_experiment,
    rank_zero_only,
)

class ConsoleLogger(Logger):
    """Simple logger that prints metrics to the console using ``logging``.

    Parameters
    ----------
    precision : int or None, optional
        If set, numerical metrics are formatted with this many decimal places.
        ``None`` disables rounding.
    fmt : {"e", "f"}, optional
        Format style for floating point numbers, either scientific (``"e"``)
        or fixed-point (``"f"``). Default is ``"f"``.
    """

    def __init__(self, precision=None, fmt="f"):
        super().__init__()
        self._step = 0
        self.precision = precision
        self.fmt = fmt
        self.logger = logging.getLogger("ConsoleLogger")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @property
    def name(self):
        return "console"

    @property
    def version(self):
        return "0"

    @property
    @rank_zero_experiment
    def experiment(self):
        return self.logger

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info("Hyperparameters:")
        for k, v in params.items():
            self.logger.info("  %s: %s", k, v)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if step is not None:
            self._step = step
        formatted = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if isinstance(v, float) and self.precision is not None:
                    value = format(v, f".{self.precision}{self.fmt}")
                else:
                    value = str(v)
            else:
                value = str(v)
            formatted.append(f"{k}={value}")
        metric_str = ", ".join(formatted)
        self.logger.info("step %s: %s", self._step, metric_str)

    @rank_zero_only
    def finalize(self, status):
        self.logger.info("Run finished with status: %s", status)

