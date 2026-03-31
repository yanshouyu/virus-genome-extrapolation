"""Configuration on probing tasks.
"""
import argparse
import dataclasses
import json
import os
import time

NON_TRAINING_ARGS = [
    "organism", 
    "run_id", 
    "prune_full_model",
    "hidden_dim",
    "base_model",
]

@dataclasses.dataclass
class Config:
    """Configurature for model training with command-line argument parsing"""
    # task config
    organism: str = ""

    # model config
    hidden_dim: int = 512
    base_model: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

    # training config
    run_id: str = f"{int(time.time() * 1000)}"
    output_dir: str = f"trainer_{run_id}"
    per_device_train_batch_size: int = 500
    per_device_eval_batch_size: int = 500
    num_train_epochs: int = 50
    learning_rate: float = 3e-4

    # logging & monitoring config
    report_to: str = "tensorboard"
    logging_strategy: str = "steps"
    logging_steps: int = 100    # frequent tracking for experiments
    disable_tqdm: bool = True    # disable tqdm for easy slurm output

    # eval config
    eval_strategy: str = "steps"

    # checkpoint
    save_strategy: str = "steps"
    save_steps: int = 300
    prune_full_model: bool = False    # save only classifier if true

    # reproducibility
    seed: int = 42
    data_seed: int = 42

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--organism", help="data source: human / virus")
        parser.add_argument(
            "--hidden-dim", type=int, default=512, help="Hidden dim of MLP-1 classifier [512]"
        )
        parser.add_argument(
            "--base-model",
            type=str,
            default="InstaDeepAI/nucleotide-transformer-500m-human-ref",
            help="HuggingFace base model name [InstaDeepAI/nucleotide-transformer-500m-human-ref]",
        )
        parser.add_argument(
            "--epochs", type=int, default=50, help="Training epochs [50]"
        )
        parser.add_argument(
            "--lr", type=float, default=3e-4, help="learning rate [3e-4]"
        )
        args = parser.parse_args()

        self.organism = args.organism
        if self.organism == "virus":
            self.per_device_train_batch_size //= 10
            self.per_device_eval_batch_size //= 10
            self.logging_steps //= 10
            self.save_steps //= 10

        self.output_dir = "_".join([self.output_dir, self.organism])
        self.hidden_dim = args.hidden_dim
        self.base_model = args.base_model
        self.num_train_epochs = args.epochs
        self.learning_rate = args.lr

    def get_training_args(self):
        training_args = dataclasses.asdict(self)
        for k in NON_TRAINING_ARGS:
            _ = training_args.pop(k)
        return training_args

    def save_config(self):
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)
