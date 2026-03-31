"""MLM domain adaptation pretraining entrypoint.

Usage (local dry-run / SLURM):
    python -m nt_domain_adapt.pretrain [options]
    nt-domain-adapt [options]              # after `pip install -e .`

All options default to sensible values; see AdaptConfig and --help for details.
Training is designed to run on a SLURM-managed GPU node — do not run on login nodes.
"""
import json
import os

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from nt_domain_adapt.callbacks import (
    BestModelExportCallback,
    PerplexityLoggingCallback,
    RunMetadataCallback,
)
from nt_domain_adapt.configs import AdaptConfig
from nt_domain_adapt.corpus import (
    build_corpus,
    corpus_hash,
    corpus_report,
    tokenize_and_split,
)

# Place TensorBoard logs alongside the existing probe training logs
os.environ["TENSORBOARD_LOGGING_DIR"] = "./tensorboard_logging"


def main() -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    cfg = AdaptConfig()
    cfg.parse_args()
    print(f"Output dir : {cfg.output_dir}")
    print(f"Base model : {cfg.base_model}")
    cfg.save_config()

    # ── Corpus preparation ───────────────────────────────────────────────────
    print("\nBuilding corpus …")
    windows, manifest = build_corpus(
        cfg.gb_dir,
        window_nt=cfg.window_nt,
        stride_nt=cfg.stride_nt,
        max_ambiguous_frac=cfg.max_ambiguous_frac,
    )
    print(f"  {len(windows)} windows from {len(manifest)} genomes")

    c_hash = corpus_hash(windows)
    print(f"  Corpus hash : {c_hash}")

    # ── Tokenisation ─────────────────────────────────────────────────────────
    print("\nLoading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)

    print("Tokenising and splitting …")
    dataset = tokenize_and_split(
        windows,
        tokenizer,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )
    print(f"  Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}")

    # ── Corpus report ─────────────────────────────────────────────────────────
    report = corpus_report(windows, manifest, dataset)
    report_path = os.path.join(cfg.output_dir, "corpus_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"  Corpus report written to {report_path!r}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading model …")
    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model, trust_remote_code=True)

    # ── Trainer setup ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(**cfg.get_training_args())

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability,
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
        PerplexityLoggingCallback(),
        BestModelExportCallback(cfg.output_dir),
        RunMetadataCallback(cfg, c_hash),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training …")
    trainer.train()

    with open(os.path.join(cfg.output_dir, "trainer_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
