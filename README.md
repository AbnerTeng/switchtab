# switchtab

A more comprehensive unofficial implementation of "SwitchTab: Switched Autoencoders Are Effective Tabular Learners"

## Installation

### Install packages via `uv`

Install [uv](https://github.com/astral-sh/uv) for package management

```bash
uv init && uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Install packages via `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install torch manually

Install torch with compatible cuda version manually from [source](https://pytorch.org/get-started/previous-versions/)

* Make sure to start with `uv pip install` if you are using `uv` for package management.

## Usage

Utilize wandb for logging and tracking experiments.

```bash
pip install wandb
wandb login
```

Execute pre-training script

```bash
python -m src.main --config_path config/settings.yaml --device cuda:0 --ssl_only --wandb_track  --project-name switchtab --run-name pretrain-ssl-only
```

or

```bash
uv run python -m src.main --config_path config/settings.yaml --device cpu --ssl_only --wandb_track --project-name switchtab --run-name pretrain-ssl-only
```

if you have `uv` installed and activated.

## Further Issue

* [ ] finetuning code
* [ ] experiments from original paper
