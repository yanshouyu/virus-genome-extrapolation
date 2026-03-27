from transformers import Trainer, TrainingArguments, default_data_collator
import dataclasses
from baculo_tss_probe.callbacks import SaveHeadCallback
from baculo_tss_probe.data import prep_ds
from baculo_tss_probe.models import load_pretrained_nt
from baculo_tss_probe.configs import Config

def main():
    "Main training function"
    cfg = Config()
    cfg.parse_args()
    print(f"Output dir: {cfg.output_dir}")

    model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    tokenizer, model = load_pretrained_nt(model_name)
    train_ds, eval_ds = prep_ds(cfg.organism, tokenizer)

    training_args = TrainingArguments(
        **cfg.get_training_args()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        # set prune_full_model if we want to delete big base model files
        callbacks=[SaveHeadCallback(prune_full_model=cfg.prune_full_model)],

    )
    trainer.train()


if __name__ == "__main__":
    main()