"""Train on training data, select by validation loss, evaluate final test once."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from baculo_tss_probe.callbacks import SaveHeadCallback
from baculo_tss_probe.data import prep_ds
from baculo_tss_probe.models import load_pretrained_nt
from baculo_tss_probe.configs import Config


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall': recall_score(labels, preds, average='binary', zero_division=0),
        'matthews_correlation': matthews_corrcoef(labels, preds),
    }


def main():
    cfg = Config()
    cfg.parse_args()
    # The head and any missing model parameters are created before Trainer.
    set_seed(cfg.seed)
    tokenizer, model = load_pretrained_nt(cfg.base_model, hidden_dim=cfg.hidden_dim,
                                           revision=cfg.model_revision)
    cfg.model_revision = model.config.backbone_revision
    dataset = prep_ds(cfg.organism, tokenizer, data_dir=cfg.data_dir, seed=cfg.data_seed)
    cfg.save_config()
    output = Path(cfg.output_dir)
    tokenizer.save_pretrained(output)
    model.config.save_pretrained(output / 'model-config')
    root = Path(__file__).resolve().parents[2]
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, capture_output=True, text=True)
    versions = {package: importlib.metadata.version(package)
                for package in ('torch', 'transformers', 'datasets', 'numpy', 'pandas', 'accelerate')}
    manifest = root / cfg.data_dir / 'manifest.json'
    metadata = {'git_commit': git.stdout.strip() if git.returncode == 0 else None,
                'git_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                'packages': versions, 'model_revision': cfg.model_revision,
                'python': sys.version,
                'pooling': model.config.pooling, 'seed': cfg.seed,
                'dataset_fingerprints': {name: records._fingerprint for name, records in dataset.items()},
                'dataset_manifest': json.loads(manifest.read_text())
                if cfg.organism == 'virus' and manifest.is_file() else None}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if cfg.prune_full_model and cfg.load_best_model_at_end:
        raise ValueError('Cannot prune full checkpoints while loading the best model.')
    trainer = Trainer(
        model=model, args=TrainingArguments(**cfg.get_training_args()),
        train_dataset=dataset['train'], eval_dataset=dataset['validation'],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[SaveHeadCallback(prune_full_model=cfg.prune_full_model)],
    )
    trainer.train()
    trainer.save_model(str(output / 'final-model'))
    # Test metrics never drive checkpoint/epoch selection.
    test_output = trainer.predict(dataset['test'], metric_key_prefix='test')
    (output / 'test_metrics.json').write_text(json.dumps(test_output.metrics, indent=2) + '\n')
    np.savez_compressed(output / 'test_predictions.npz', logits=test_output.predictions,
                        labels=test_output.label_ids)
    (output / 'trainer_history.json').write_text(json.dumps(trainer.state.log_history, indent=2) + '\n')


if __name__ == '__main__':
    main()
