#!/usr/bin/env python
import wandb

from autorag.config import SWEEP_CONFIG
from autorag.evaluate import evaluate

if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="autorag-demo")
    wandb.agent(sweep_id, evaluate)
