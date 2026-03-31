"""Callbacks for NT domain adaptation pretraining."""
import dataclasses
import json
import math
import os
import subprocess

from transformers import TrainerCallback


class PerplexityLoggingCallback(TrainerCallback):
    """Adds ``eval_perplexity = exp(eval_loss)`` to every log event that contains
    ``eval_loss``, so it appears in TensorBoard alongside the raw loss.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            # Cap exponent to avoid overflow on early unstable steps
            logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20.0))


class BestModelExportCallback(TrainerCallback):
    """Copies the best checkpoint (loaded by the Trainer when
    ``load_best_model_at_end=True``) to ``models/<run_dir_name>/`` after
    training completes.
    """

    def __init__(self, run_output_dir: str) -> None:
        self.run_output_dir = run_output_dir

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")

        export_dir = os.path.join("models", os.path.basename(self.run_output_dir))
        os.makedirs(export_dir, exist_ok=True)

        if model is not None:
            model.save_pretrained(export_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(export_dir)

        export_info = {
            "best_model_checkpoint": state.best_model_checkpoint,
            "best_metric_value": state.best_metric,
            "export_path": export_dir,
        }
        with open(os.path.join(export_dir, "export_info.json"), "w", encoding="utf-8") as f:
            json.dump(export_info, f, indent=4)

        print(f"Best model exported to {export_dir!r} (best_metric={state.best_metric})")


class RunMetadataCallback(TrainerCallback):
    """Writes ``run_metadata.json`` to the run output directory at the end of
    training.  Contains the full config, corpus hash, and the git commit SHA so
    the run is fully reproducible.
    """

    def __init__(self, cfg, corpus_hash: str) -> None:
        self.cfg = cfg
        self.corpus_hash = corpus_hash

    def on_train_end(self, args, state, control, **kwargs):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            commit_hash = result.stdout.strip()
        except Exception:
            commit_hash = "unknown"

        # Summarise final metrics from log history
        final_train_loss = next(
            (e["loss"] for e in reversed(state.log_history) if "loss" in e), None
        )
        final_eval_loss = next(
            (e["eval_loss"] for e in reversed(state.log_history) if "eval_loss" in e), None
        )

        metadata = {
            "config": dataclasses.asdict(self.cfg),
            "corpus_hash": self.corpus_hash,
            "commit_hash": commit_hash,
            "best_model_checkpoint": state.best_model_checkpoint,
            "best_metric_value": state.best_metric,
            "total_train_steps": state.global_step,
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "final_eval_perplexity": (
                math.exp(min(final_eval_loss, 20.0)) if final_eval_loss is not None else None
            ),
        }

        output_path = os.path.join(self.cfg.output_dir, "run_metadata.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Run metadata written to {output_path!r}")
