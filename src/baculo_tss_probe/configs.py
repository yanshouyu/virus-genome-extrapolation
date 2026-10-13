"""Configuration for frozen-backbone probing."""
import argparse
import dataclasses
import json
import os
import time
import uuid

NON_TRAINING_ARGS = ('organism', 'run_id', 'prune_full_model', 'hidden_dim', 'base_model',
                     'model_revision', 'data_dir')


@dataclasses.dataclass
class Config:
    organism: str = ''
    hidden_dim: int = 512
    base_model: str = 'InstaDeepAI/nucleotide-transformer-500m-human-ref'
    model_revision: str | None = None
    data_dir: str = 'data/processed/clean_v2'
    run_id: str = dataclasses.field(default_factory=lambda: f'{time.time_ns()}_{uuid.uuid4().hex[:6]}')
    output_dir: str = ''
    per_device_train_batch_size: int = 500
    per_device_eval_batch_size: int = 500
    num_train_epochs: int = 50
    learning_rate: float = 3e-4
    report_to: str = 'tensorboard'
    logging_strategy: str = 'steps'
    logging_steps: int = 100
    disable_tqdm: bool = True
    eval_strategy: str = 'steps'
    save_strategy: str = 'steps'
    save_steps: int = 300
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'loss'
    greater_is_better: bool = False
    prune_full_model: bool = False
    seed: int = 42
    data_seed: int = 42

    def parse_args(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--organism', required=True, choices=('human', 'virus'))
        parser.add_argument('--hidden-dim', type=int, default=self.hidden_dim)
        parser.add_argument('--base-model', default=self.base_model)
        parser.add_argument('--model-revision', default=self.model_revision)
        parser.add_argument('--data-dir', default=self.data_dir)
        parser.add_argument('--output-dir')
        parser.add_argument('--epochs', type=int, default=self.num_train_epochs)
        parser.add_argument('--lr', type=float, default=self.learning_rate)
        parser.add_argument('--seed', type=int, default=self.seed)
        args = parser.parse_args(argv)
        if args.hidden_dim <= 0 or args.epochs <= 0 or args.lr <= 0:
            parser.error('hidden dimension, epochs and learning rate must be positive.')
        self.organism = args.organism
        if self.organism == 'virus':
            self.per_device_train_batch_size //= 10
            self.per_device_eval_batch_size //= 10
            self.logging_steps //= 10
            self.save_steps //= 10
        self.output_dir = args.output_dir or f'runs/probe_{self.run_id}_{self.organism}'
        self.hidden_dim = args.hidden_dim
        self.base_model = args.base_model
        self.model_revision = args.model_revision
        self.data_dir = args.data_dir
        self.num_train_epochs = args.epochs
        self.learning_rate = args.lr
        self.seed = self.data_seed = args.seed

    def get_training_args(self):
        training_args = dataclasses.asdict(self)
        for key in NON_TRAINING_ARGS:
            training_args.pop(key)
        return training_args

    def save_config(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'config.json'), 'w', encoding='utf-8') as handle:
            json.dump(dataclasses.asdict(self), handle, indent=2)
