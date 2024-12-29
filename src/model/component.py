from typing import Literal

import torch
import torch.nn as nn


def feature_corruption(
    x: torch.Tensor,
    device: str,
    corruption_type: Literal["noise", "masking", "none"] = "noise",
    corruption_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Feature corruption function

    Corruption types
        - noise: Add noise to part of the input tensor
        - masking: Mask part of the input tensor
        - none: Do nothing
    """
    num_features = x.size(1)
    num_corrupt = int(num_features * corruption_ratio)

    corrupt_indices = torch.randperm(num_features, device=device)[:num_corrupt]

    if corruption_type == "noise":
        noise = torch.randn((x.size(0), num_corrupt), device=device) * 0.1
        x[:, corrupt_indices] = noise

    elif corruption_type == "masking":
        x[:, corrupt_indices] = 0

    else:
        pass

    return x


class TFMEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2):
        super().__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_layers(x)


class Projector(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class Predictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ReconstructLoss(nn.Module):
    """
    How to make sure that x1s is similar to x1r...
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self,
        org_x1: torch.Tensor,
        org_x2: torch.Tensor,
        x1r: torch.Tensor,
        x1s: torch.Tensor,
        x2r: torch.Tensor,
        x2s: torch.Tensor,
    ) -> torch.Tensor:
        loss1 = self.loss(org_x1, x1r)
        loss2 = self.loss(org_x2, x2r)
        loss3 = self.loss(org_x1, x1s)
        loss4 = self.loss(org_x2, x2s)

        return loss1 + loss2 + loss3 + loss4
