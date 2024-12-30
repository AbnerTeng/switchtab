from typing import Any, Dict

from argparse import ArgumentParser, Namespace
import wandb
import torch
from torch import nn
from torch.optim import RMSprop

from .trainer import SwitchTabTrainer
from .utils import load_config
from .data.dataset import load_training_data
from .model.switch_tab import (
    DataEncoder,
    SSLFramework,
)
from .model.component import ReconstructLoss
from .model.mlp_label import LabelPretrainer


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/settings.yaml")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--ssl_only", action="store_true")
    parser.add_argument("--wandb_track", action="store_true")
    parser.add_argument("--project-name", type=str, default="switchtab")
    parser.add_argument("--run-name", type=str, default="test")

    return parser.parse_args()


def init_wandb(args: Namespace, device: str, settings: Dict[str, Any]) -> None:
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config={
            "device": device,
            "encoder_config": settings["encoder"],
            "optimizer": settings["optimizer"],
            "trainer": settings["trainer"],
            "data": settings["data"],
            "label_pretrainer": settings["label_pretrainer"],
        },
    )


if __name__ == "__main__":
    args = get_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = args.device
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    settings = load_config(args.config_path)

    if args.wandb_track:
        init_wandb(args, device, settings)

    tr_loader1, _, col_cat_count1, label_cat_count1 = load_training_data(
        **settings["data"]
    )
    tr_loader2, _, col_cat_count2, label_cat_count2 = load_training_data(
        **settings["data"]
    )
    encoder = DataEncoder(settings["encoder"], device)
    ssl_framework = SSLFramework(
        len(col_cat_count1), 1 if label_cat_count1 == -1 else label_cat_count1
    )
    label_pretrainer = LabelPretrainer(
        len(col_cat_count1),
        1 if label_cat_count1 == -1 else label_cat_count1,
        settings["label_pretrainer"],
    )
    encoder.to(device)
    ssl_framework.to(device)
    label_pretrainer.to(device)

    if args.wandb_track:
        wandb.watch((encoder, ssl_framework, label_pretrainer), log="all", log_freq=100)

    trainer = SwitchTabTrainer(
        tr_loader1,
        tr_loader2,
        encoder,
        ssl_framework,
        label_pretrainer,
        ReconstructLoss(),
        nn.CrossEntropyLoss() if label_cat_count1 != -1 else nn.MSELoss(),
        RMSprop(
            list(encoder.parameters())
            + list(ssl_framework.parameters())
            + list(label_pretrainer.parameters()),
            **settings["optimizer"]["rmsprop"],
        ),
        settings["trainer"]["pretrain"],
    )

    if args.wandb_track:
        trainer.wandb = wandb

    print("Start training")

    if args.wandb_track:
        try:
            trainer.train(ssl_only=args.ssl_only)
        finally:
            wandb.finish()

    else:
        trainer.train(ssl_only=args.ssl_only)
