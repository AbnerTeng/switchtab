from typing import List, Tuple, Any, Dict

import torch
from torch import nn


class LabelPretrainer(nn.Module):
    """
    Args:
        input_dim (int): dimension of encoded feature.
        hidden_dims (Union[List[int], int]): Hidden layer dimensions.
        output_dim (int): Number of classes.
        n_hiddens (Optional[int]): Number of hidden layers.
        config (Dict[str, Any]): Configuration dictionary.
    """

    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        layers: List[nn.Module] = []

        if self.config["use_batch_norm"]:
            batch_norm = nn.BatchNorm1d
        else:
            batch_norm = nn.Identity

        if isinstance(self.config["hidden_dims"], int):
            self.config["hidden_dims"] = [self.config["hidden_dims"]]

        for i in range(self.config["n_hiddens"] - 1):
            layers.append(nn.Linear(self.input_dim, self.config["hidden_dims"][i]))
            layers.append(batch_norm(self.config["hidden_dims"][i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout_rate"]))
            self.input_dim = self.config["hidden_dims"][i]

        layers.append(nn.Linear(self.config["hidden_dims"][-1], self.output_dim))

        self.model = nn.Sequential(*layers)

    def forward(
        self, encoded_x1: torch.Tensor, encoded_x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_x1 = self.model(encoded_x1)
        pred_x2 = self.model(encoded_x2)

        return pred_x1, pred_x2
