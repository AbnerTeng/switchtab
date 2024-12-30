from typing import Any, Dict, Tuple

import torch
from torch import nn

from .component import (
    TFMEncoder,
    Projector,
    Decoder,
    Predictor,
    feature_corruption,
)


class DataEncoder(nn.Module):
    def __init__(self, d_model: int, device: str) -> None:
        super().__init__()
        self.encoder = TFMEncoder(d_model=d_model)
        self.device = device

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        corrupted_x1 = feature_corruption(x1, device=self.device)
        corrupted_x2 = feature_corruption(x2, device=self.device)
        encoded_x1 = self.encoder(corrupted_x1)
        encoded_x2 = self.encoder(corrupted_x2)

        return encoded_x1, encoded_x2


class SSLFramework(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.salient_projector = Projector(feature_size)
        self.mutual_projector = Projector(feature_size)
        self.decoder = Decoder(2 * feature_size, feature_size)
        self.predictor = Predictor(feature_size, num_classes)

    def forward(
        self, encoded_x1: torch.Tensor, encoded_x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        salient_x1 = self.salient_projector(encoded_x1)
        salient_x2 = self.salient_projector(encoded_x2)
        mutual_x1 = self.mutual_projector(encoded_x1)
        mutual_x2 = self.mutual_projector(encoded_x2)

        recovered_x1 = self.decoder(torch.cat([mutual_x1, salient_x1], dim=1))
        switched_x1 = self.decoder(torch.cat([mutual_x1, salient_x2], dim=1))
        recovered_x2 = self.decoder(torch.cat([mutual_x2, salient_x2], dim=1))
        switched_x2 = self.decoder(torch.cat([mutual_x2, salient_x1], dim=1))

        return recovered_x1, switched_x1, recovered_x2, switched_x2
