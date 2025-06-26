import torch
import torch.nn as nn

from .cell_model import Cell_Unet
from .nn import zero_module


class ControlNet(nn.Module):
    """Apply external control to the diffusion model."""

    def __init__(self, input_dim=2, hidden_num=None, dropout=0.1):
        super().__init__()
        if hidden_num is None:
            hidden_num = [2000, 1000, 500, 500]
        self.control_unet = Cell_Unet(input_dim, hidden_num, dropout)
        # Start from zero weights so control net does not affect inference if not trained
        zero_module(self.control_unet)

    @staticmethod
    def preprocess_control(x):
        """Convert raw control matrix to network input and mask."""
        mask = x.ne(-1).float()
        x = x.clone()
        x[x == -1] = 0
        x = torch.log1p(x)
        return x, mask

    def forward(self, control, t):
        control, mask = self.preprocess_control(control)
        out = self.control_unet(control, t)
        return out * mask
