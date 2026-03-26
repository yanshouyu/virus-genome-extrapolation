from transformers import Trainer, TrainingArguments, default_data_collator
import argparse
from baculo_tss_probe.callbacks import SaveHeadCallback
from baculo_tss_probe.data import prep_ds
from baculo_tss_probe.models import load_pretrained_nt

def train(organism, epochs):
    model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    tokenizer, model = load_pretrained_nt(model_name)
    train_ds, eval_ds = prep_ds(organism, tokenizer)

    training_args = TrainingArguments(
        output_dir="test_trainer", 
        eval_strategy='epoch', 
        save_strategy="epoch",
        num_train_epochs=epochs,
        load_best_model_at_end=False,
        remove_unused_columns=True,
        seed=42, 
        data_seed=42, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[SaveHeadCallback(prune_full_model=False)],  # set True to delete big files

    )
    trainer.train()


if __name__ == "__main__":
    # TODO: parse config args
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism", help="data source: human / virus")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    args = parser.parse_args()
    args = vars(args)
    
    # MVP: test run
    train(args['organism'], args['epochs'])