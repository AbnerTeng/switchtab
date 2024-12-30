from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from rich import print as rp
from rich.progress import track


class SwitchTabTrainer:
    def __init__(
        self,
        train_loader1: DataLoader,
        train_loader2: DataLoader,
        encoder: nn.Module,
        ssl_framework: nn.Module,
        label_pretrainer: nn.Module,
        ssl_criterion: Any,
        label_criterion: Any,
        optimizer: torch.optim.Optimizer,
        train_config: Dict[str, Any],
    ) -> None:
        self.train_loader1 = train_loader1
        self.train_loader2 = train_loader2
        self.encoder = encoder
        self.ssl_framework = ssl_framework
        self.label_pretrainer = label_pretrainer
        self.ssl_criterion = ssl_criterion
        self.label_criterion = label_criterion
        self.optimizer = optimizer
        self.train_config = train_config
        self.wandb = None

    def train(self, ssl_only: bool) -> None:
        self.encoder.train()
        self.ssl_framework.train()
        self.label_pretrainer.train()

        best_loss = np.inf

        for epoch in track(range(self.train_config["n_epochs"])):
            total_loss = torch.tensor(0.0).to(self.encoder.device)
            loss_seq = []

            for xy1_batch, xy2_batch in zip(self.train_loader1, self.train_loader2):
                x1_batch, x2_batch = (
                    xy1_batch[0].to(self.encoder.device),
                    xy2_batch[0].to(self.encoder.device),
                )
                y1_batch, y2_batch = (
                    xy1_batch[1].to(self.encoder.device),
                    xy2_batch[1].to(self.encoder.device),
                )
                encoded_x1, encoded_x2 = self.encoder(x1_batch, x2_batch)
                recovered_x1, switched_x1, recovered_x2, switched_x2 = (
                    self.ssl_framework(encoded_x1, encoded_x2)
                )

                ssl_loss = self.ssl_criterion(
                    x1_batch,
                    x2_batch,
                    recovered_x1,
                    switched_x1,
                    recovered_x2,
                    switched_x2,
                )

                if not ssl_only:
                    pred_x1, pred_x2 = self.label_pretrainer(encoded_x1, encoded_x2)
                    label_loss = self.label_criterion(
                        y1_batch.unsqueeze(1),
                        pred_x1
                    ) + self.label_criterion(
                        y2_batch.unsqueeze(1),
                        pred_x2
                    )
                    total_loss = ssl_loss + self.train_config["alpha"] * label_loss
                else:
                    total_loss = ssl_loss

                loss_seq.append(total_loss.item())

                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": total_loss.item(),
                            "ssl_loss": ssl_loss.item(),
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if (epoch + 1) % self.train_config["print_interval"] == 0:
                rp(f"Epoch: {epoch + 1}, Avg Loss: {np.mean(loss_seq):.4f}")

            if np.mean(loss_seq) < best_loss:
                best_loss = np.mean(loss_seq)
                rp(f"Saving best model on epoch: {epoch + 1} | Loss: {best_loss:.4f}")
                ssl_only_str: str = "ssl" if ssl_only else "ssl_label"
                torch.save(
                    {
                        "encoder": self.encoder.state_dict(),
                        "ssl_framework": self.ssl_framework.state_dict(),
                        "label_pretrainer": self.label_pretrainer.state_dict(),
                    },
                    f"model/best_model_{ssl_only_str}.pth",
                )

    def finetune(self) -> None:
        pass
