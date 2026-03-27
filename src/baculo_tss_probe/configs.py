"""Configuration on probing tasks.
"""
import argparse
import dataclasses
import time

@dataclasses.dataclass
class Config:
    """Configurature for model training with command-line argument parsing"""
    # task config
    organism: str = ""

    # training config
    run_id: str = f"{int(time.time() * 1000)}"
    output_dir: str = f"trainer_{run_id}"
    per_device_train_batch_size: int = 500
    per_device_eval_batch_size: int = 500
    num_train_epochs: int = 50
    learning_rate: float = 3e-4

    # logging & monitoring config
    logging_strategy: str = "steps"
    logging_steps: int = 500
    disable_tqdm: bool = True    # disable tqdm for easy slurm output

    # eval config
    eval_strategy: str = "steps"

    # checkpoint
    save_strategy: str = "epoch"
    prune_full_model: bool = False    # save only classifier if true

    # reproducibility
    seed: int = 42
    data_seed: int = 42

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--organism", help="data source: human / virus")
        parser.add_argument(
            "--epochs", type=int, default=50, help="Training epochs [50]"
        )
        parser.add_argument(
            "--lr", type=float, default=3e-4, help="learning rate [3e-4]"
        )
        args = parser.parse_args()

        self.organism = args.organism
        self.output_dir = "_".join([self.output_dir, self.organism])
        self.num_train_epochs = args.epochs
        self.learning_rate = args.lr

    def get_training_args(self):
        training_args = dataclasses.asdict(self)
        for k in ["organism", "run_id", "prune_full_model"]:
            _ = training_args.pop(k)
        return training_args

