"""Configuration for NT MLM domain adaptation pretraining."""
import argparse
import dataclasses
import json
import os
import time

NON_TRAINING_ARGS = {
    "base_model",
    "run_id",
    "gb_dir",
    "window_nt",
    "stride_nt",
    "max_ambiguous_frac",
    "val_frac",
    "mlm_probability",
    "early_stopping_patience",
}

_run_id = f"{int(time.time() * 1000)}"


@dataclasses.dataclass
class AdaptConfig:
    """Configuration for MLM domain adaptation with command-line argument parsing."""

    # ── data ────────────────────────────────────────────────────────────────
    base_model: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    gb_dir: str = "data/raw/baculovirus/Baculoviridae"
    window_nt: int = 6144      # nucleotides per window (= 1024 × 6-mer tokens)
    stride_nt: int = 3072      # stride between windows
    max_ambiguous_frac: float = 0.05   # drop window if non-ACGT fraction exceeds this
    val_frac: float = 0.10     # fraction of windows held out for validation

    # ── MLM ─────────────────────────────────────────────────────────────────
    mlm_probability: float = 0.15

    # ── run identity ─────────────────────────────────────────────────────────
    run_id: str = _run_id
    output_dir: str = f"runs/domain_adapt_{_run_id}"

    # ── optimisation ─────────────────────────────────────────────────────────
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 32   # effective batch = 4 × 32 = 128
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True

    # ── early stopping ────────────────────────────────────────────────────────
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # ── logging & monitoring ──────────────────────────────────────────────────
    report_to: str = "tensorboard"
    logging_strategy: str = "steps"
    logging_steps: int = 50
    disable_tqdm: bool = True

    # ── eval & checkpoint ─────────────────────────────────────────────────────
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    prediction_loss_only: bool = True   # avoids materialising huge MLM logits

    # ── reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    data_seed: int = 42

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(
            description="Baculovirus NT MLM domain adaptation"
        )
        parser.add_argument("--base-model", default=self.base_model,
                            help="HF model name or path [%(default)s]")
        parser.add_argument("--gb-dir", default=self.gb_dir,
                            help="Directory of GenBank (.gb) files [%(default)s]")
        parser.add_argument("--window-nt", type=int, default=self.window_nt,
                            help="Window size in nucleotides [%(default)s]")
        parser.add_argument("--stride-nt", type=int, default=self.stride_nt,
                            help="Stride between windows in nucleotides [%(default)s]")
        parser.add_argument("--mlm-probability", type=float, default=self.mlm_probability,
                            help="Fraction of tokens masked per sample [%(default)s]")
        parser.add_argument("--epochs", type=int, default=self.num_train_epochs,
                            help="Maximum training epochs [%(default)s]")
        parser.add_argument("--lr", type=float, default=self.learning_rate,
                            help="Learning rate [%(default)s]")
        parser.add_argument(
            "--effective-batch-size", type=int, default=128,
            help="Target effective batch size; sets gradient_accumulation_steps automatically [128]"
        )
        parser.add_argument("--early-stopping-patience", type=int,
                            default=self.early_stopping_patience,
                            help="Epochs without eval_loss improvement before stopping [%(default)s]")
        parser.add_argument("--output-dir", default="",
                            help="Override default output directory")
        args = parser.parse_args()

        self.base_model = args.base_model
        self.gb_dir = args.gb_dir
        self.window_nt = args.window_nt
        self.stride_nt = args.stride_nt
        self.mlm_probability = args.mlm_probability
        self.num_train_epochs = args.epochs
        self.learning_rate = args.lr
        self.gradient_accumulation_steps = (
            args.effective_batch_size // self.per_device_train_batch_size
        )
        self.early_stopping_patience = args.early_stopping_patience
        if args.output_dir:
            self.output_dir = args.output_dir

    def get_training_args(self) -> dict:
        """Return fields that map directly to TrainingArguments kwargs."""
        d = dataclasses.asdict(self)
        for k in NON_TRAINING_ARGS:
            d.pop(k, None)
        return d

    def save_config(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)
