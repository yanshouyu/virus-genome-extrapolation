import csv
import json
import shutil
from pathlib import Path

import datasets
import pytest
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.models.esm.configuration_esm import EsmConfig
from baculo_tss_probe.configs import Config
from baculo_tss_probe.data import load_dataset_dict, prep_ds
from baculo_tss_probe.models import SequenceClassification, SequenceClassificationConfig
import baculo_tss_probe.train as training

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
REVISION = '06615c1660c892fc199840c18123f8385b3542a8'


def test_selection_never_tracks_an_unsaved_validation_candidate(tmp_path):
    """An unsaved step-1 improvement must not block selecting saved step 2."""
    cfg = Config(organism='virus', output_dir=str(tmp_path), num_train_epochs=3,
                 per_device_train_batch_size=4, per_device_eval_batch_size=4,
                 logging_steps=1, save_steps=2, report_to='none')
    cfg.save_config()
    data = datasets.Dataset.from_dict({'x': [[1.], [2.], [3.], [4.]], 'labels': [0, 1, 0, 1]})
    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2)
        def forward(self, x, labels=None):
            logits = self.linear(x)
            return {'loss': torch.nn.functional.cross_entropy(logits, labels), 'logits': logits}
    observed_steps = []
    class ControlledValidationTrainer(Trainer):
        def evaluate(self, *args, **kwargs):
            result = super().evaluate(*args, **kwargs)
            observed_steps.append(self.state.global_step)
            result['eval_loss'] = self.state.global_step / 10
            return result
    trainer = ControlledValidationTrainer(model=TinyClassifier(),
                args=TrainingArguments(**cfg.get_training_args()), train_dataset=data, eval_dataset=data)
    trainer.train()
    assert 1 not in observed_steps and 2 in observed_steps
    assert trainer.state.best_global_step == 2
    assert trainer.state.best_model_checkpoint.endswith('checkpoint-2')
    assert (Path(trainer.state.best_model_checkpoint)/'model.safetensors').is_file()
    assert json.loads((tmp_path/'config.json').read_text())['eval_steps'] == 2


def test_configuration_ids_and_required_organism():
    first, second = Config(), Config()
    assert first.run_id != second.run_id
    first.parse_args(['--organism', 'virus', '--seed', '123'])
    assert first.seed == first.data_seed == 123
    assert first.output_dir.startswith('runs/')
    assert 'data_dir' not in first.get_training_args()
    with pytest.raises(SystemExit):
        second.parse_args([])


def test_virus_loader_has_three_splits_and_rejects_duplicate_samples(tmp_path):
    dataset = load_dataset_dict('virus')
    assert [len(dataset[s]) for s in ('train', 'validation', 'test')] == [233, 77, 78]
    for name in ('train', 'validation', 'test'):
        shutil.copy(ROOT/f'data/processed/clean_v2/{name}.csv', tmp_path/f'{name}.csv')
    with (tmp_path/'test.csv').open('a') as handle:
        handle.write((tmp_path/'train.csv').read_text().splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='Duplicate sample'):
        load_dataset_dict('virus', data_dir=tmp_path)


def test_tokenization_retains_labels_and_drops_metadata():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION,
                                              trust_remote_code=True, local_files_only=True)
    dataset = prep_ds('virus', tokenizer)
    assert set(dataset) == {'train', 'validation', 'test'}
    assert 'labels' in dataset['train'].column_names
    assert not {'name', 'sequence', 'group'} & set(dataset['train'].column_names)
    assert {int(record['labels']) for record in dataset['train']} == {0, 1}


def test_synthetic_training_selects_validation_and_predicts_test_once(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION,
                                              trust_remote_code=True, local_files_only=True)
    def ds():
        return datasets.Dataset.from_dict({'input_ids': [[0, 5, 6, 2], [0, 9, 2]],
                                           'attention_mask': [[1, 1, 1, 1], [1, 1, 1]],
                                           'labels': [0, 1]}).with_format('torch')
    data = datasets.DatasetDict(train=ds(), validation=ds(), test=ds())
    cfg = Config(organism='virus', output_dir=str(tmp_path), seed=123, data_seed=123,
                 per_device_train_batch_size=2, per_device_eval_batch_size=2,
                 num_train_epochs=2, logging_steps=1, save_steps=1, report_to='none')
    cfg.parse_args = lambda: None
    monkeypatch.setattr(training, 'Config', lambda: cfg)
    observations = {'predict_calls': 0}
    def load(*args, **kwargs):
        # Verify main seeded before initialization, rather than waiting for Trainer.
        assert torch.initial_seed() == 123
        ecfg = EsmConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, pad_token_id=1,
                         mask_token_id=2, token_dropout=False, position_embedding_type='rotary')
        model = SequenceClassification(SequenceClassificationConfig(
            backbone_config=ecfg.to_dict(), backbone_has_pooler=False, hidden_dim=8, pooling='cls'))
        observations['model'] = model
        observations['original_encoder'] = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        return tokenizer, model
    monkeypatch.setattr(training, 'load_pretrained_nt', load)
    monkeypatch.setattr(training, 'prep_ds', lambda *args, **kwargs: data)
    class CheckedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            assert kwargs['eval_dataset'] is data['validation']
            super().__init__(*args, **kwargs)
        def predict(self, test_dataset, *args, **kwargs):
            assert test_dataset is data['test']
            observations['predict_calls'] += 1
            return super().predict(test_dataset, *args, **kwargs)
    monkeypatch.setattr(training, 'Trainer', CheckedTrainer)
    training.main()
    assert observations['predict_calls'] == 1
    for key, value in observations['original_encoder'].items():
        assert torch.equal(value, observations['model'].backbone.state_dict()[key])
    assert (tmp_path/'test_metrics.json').is_file()
    assert (tmp_path/'test_predictions.npz').is_file()
    assert (tmp_path/'final-model/tokenizer_config.json').is_file()
    metadata = json.loads((tmp_path/'run_metadata.json').read_text())
    assert metadata['seed'] == 123 and metadata['pooling'] == 'cls'
    assert all((checkpoint/'tokenizer_config.json').is_file() for checkpoint in tmp_path.glob('checkpoint-*'))
