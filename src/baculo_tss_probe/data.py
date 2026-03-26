import os
import datasets

def load_dataset_dict(name: str) -> datasets.dataset_dict.DatasetDict:
    if name == "human":
        dataset = datasets.load_dataset(
            "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
        )
        task = "promoter_all"
        return dataset.filter(lambda example: example["task"] == task)
    elif name == "virus":
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_path, "../../data/processed")
        data_files = {
            "train": os.path.join(data_dir, "train.csv"),
            "test": os.path.join(data_dir, "test.csv")
        }
        return datasets.load_dataset("csv", data_files=data_files)
    else:
        raise ValueError(f"{name} not specified, (human, virus)")

# TODO: methods for sample a given sized subset

def prep_ds(name, tokenizer=None):
    "prepare train_ds and val_ds for transformers.Trainer"
    dataset_dict = load_dataset_dict(name)

    if tokenizer is None:
        raise ValueError("tokenizer required")
    
    def tokenize_fn(examples):
        return tokenizer(examples['sequence'])
    
    # Tokenize all splits
    tokenized_ds = dataset_dict.map(
        tokenize_fn,
        batched=True,
        num_proc=4  # parallel processing
    )
    tokenized_ds = (
        tokenized_ds
        .rename_column("label", "labels")
        .with_format(type="torch")
    )
    return tokenized_ds['train'], tokenized_ds['test']
