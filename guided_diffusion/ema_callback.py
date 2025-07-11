import pytorch_lightning as pl
from .nn import update_ema


class EMACallback(pl.Callback):
    """Apply exponential moving average to model parameters."""

    def __init__(self, ema_rate: float = 0.9999):
        super().__init__()
        self.ema_rate = ema_rate
        self.ema_params = None

    def on_train_start(self, trainer, pl_module):
        self.ema_params = [p.clone().detach() for p in pl_module.model.parameters()]
        for p in self.ema_params:
            p.requires_grad = False

    def on_after_backward(self, trainer, pl_module):
        update_ema(self.ema_params, pl_module.model.parameters(), rate=self.ema_rate)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state_dict"] = [p.clone() for p in self.ema_params]

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        ema_state = checkpoint.get("ema_state_dict")
        if not ema_state:
            return
        # lazily create EMA params in case on_train_start hasn't run yet
        if self.ema_params is None:
            self.ema_params = [p.clone().detach() for p in pl_module.model.parameters()]
            for p in self.ema_params:
                p.requires_grad = False
        for p, cp in zip(self.ema_params, ema_state):
            p.copy_(cp)

