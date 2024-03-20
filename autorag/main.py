#!/usr/bin/env python
import time

from loguru import logger
import wandb

from autorag.config import SWEEP_CONFIG
from autorag.evaluate import evaluate

if __name__ == "__main__":
    start = time.time()

    sweep_id = wandb.sweep(SWEEP_CONFIG, project="autorag-demo")
    wandb.agent(sweep_id, evaluate)

    end = time.time()
    logger.info(f"Whole sweep took: {end - start}")
