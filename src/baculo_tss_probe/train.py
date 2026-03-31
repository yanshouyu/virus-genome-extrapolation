from transformers import Trainer, TrainingArguments, default_data_collator
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score
from baculo_tss_probe.callbacks import SaveHeadCallback
from baculo_tss_probe.data import prep_ds
from baculo_tss_probe.models import load_pretrained_nt
from baculo_tss_probe.configs import Config

# set tensorboard logging path relative to currend working dir
os.environ["TENSORBOARD_LOGGING_DIR"] = "./tensorboard_logging"

def compute_metrics(eval_pred):
    """Compute classification metrics from model logits and labels."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "matthews_correlation": matthews_corrcoef(labels, preds),
    }

def main():
    "Main training function"
    cfg = Config()
    cfg.parse_args()
    print(f"Output dir: {cfg.output_dir}")
    print(f"Base model: {cfg.base_model}")
    cfg.save_config()

    model_name = cfg.base_model
    tokenizer, model = load_pretrained_nt(model_name, hidden_dim=cfg.hidden_dim)
    train_ds, eval_ds = prep_ds(cfg.organism, tokenizer)

    training_args = TrainingArguments(
        **cfg.get_training_args()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        # set prune_full_model if we want to delete big base model files
        callbacks=[SaveHeadCallback(prune_full_model=cfg.prune_full_model)],
    )
    trainer.train()

    with open(os.path.join(cfg.output_dir, "trainer_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)


if __name__ == "__main__":
    main()