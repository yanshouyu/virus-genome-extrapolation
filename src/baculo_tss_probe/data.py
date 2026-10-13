"""Load separate training, validation and final-test data."""
import csv
from pathlib import Path
import datasets
from baculo_tss_probe.splitting import cross_split_violations, read_known_tss, label_violations


def load_dataset_dict(name, data_dir=None, seed=42):
    if name == 'human':
        dataset = datasets.load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks_revised')
        dataset = dataset.filter(lambda example: example['task'] == 'promoter_all')
        if 'validation' not in dataset:
            # Official test is untouched. Coordinates are unavailable here;
            # this stratified validation split is not a genomic-group guarantee.
            train = dataset['train'].class_encode_column('label')
            split = train.train_test_split(test_size=0.2, seed=seed, stratify_by_column='label')
            dataset = datasets.DatasetDict(train=split['train'], validation=split['test'],
                                           test=dataset['test'])
        return dataset
    if name != 'virus':
        raise ValueError('Expected organism human or virus.')
    root = Path(__file__).resolve().parents[2]
    directory = Path(data_dir) if data_dir else root / 'data/processed/clean_v2'
    if not directory.is_absolute():
        directory = root / directory
    files = {split: directory / f'{split}.csv' for split in ('train', 'validation', 'test')}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Separate train/validation/test files required: ' + ', '.join(missing))
    rows = {split: list(csv.DictReader(path.open())) for split, path in files.items()}
    if any(not records or any('group' not in row for row in records) for records in rows.values()):
        raise ValueError('Each virus split must contain rows with group identifiers.')
    identifiers = [row['name'] for records in rows.values() for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate sample identifiers across dataset.')
    violations = cross_split_violations(rows, root / 'data/interim/tss_seq_range.csv')
    if violations:
        raise ValueError(f'Dataset failed leakage checks: {violations[:3]}')
    errors = label_violations([row for records in rows.values() for row in records],
                             read_known_tss(root / 'data/raw/baculovirus/Baculovirus_TSS.csv'))
    if errors:
        raise ValueError(f'Dataset has inconsistent TSS labels: {errors[:3]}')
    return datasets.load_dataset('csv', data_files={key: str(value) for key, value in files.items()})


def prep_ds(name, tokenizer=None, data_dir=None, seed=42):
    """Tokenize all splits without using the final test for model selection."""
    if tokenizer is None:
        raise ValueError('tokenizer required')
    dataset = load_dataset_dict(name, data_dir=data_dir, seed=seed)
    prepared = {}
    for split, records in dataset.items():
        prepared[split] = records.map(
            lambda examples: tokenizer(examples['sequence'], truncation=False),
            batched=True,
            remove_columns=[column for column in records.column_names if column != 'label'],
        ).rename_column('label', 'labels').with_format('torch')
    return datasets.DatasetDict(prepared)
