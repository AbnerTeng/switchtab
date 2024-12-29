# switchtab

Unofficial implementation of "SwitchTab: Switched Autoencoders Are Effective Tabular Learners"

## Installation

### Install torch 2.2.1

1. Install torch 2.2.1 manually from [source](https://pytorch.org/get-started/previous-versions/)

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

## Usage

Utilize wandb for logging and tracking experiments.

```bash
pip install wandb
wandb login
```

Execute pre-training script

```bash
python -m src.main --config_path config/settings.yaml --ssl_only --project-name switchtab --run-name pretrain-ssl-only
```

or

```bash
uv run python -m src.main --config_path config/settings.yaml --ssl_only --project-name switchtab --run-name pretrain-ssl-only
```

if you have `uv` installed and activated.

## Further Issue

[ ] save model checkpoint
[ ] finetuning code
[ ] experiments from original paper
