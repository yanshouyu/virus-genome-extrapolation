"""Frozen Nucleotide Transformer encoder with an explicit sequence representation."""
import importlib
import inspect
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers.pytorch_utils as _pt_utils

DEFAULT_REVISIONS = {
    'InstaDeepAI/nucleotide-transformer-500m-human-ref':
        'f87b5d7233295242e79c951873d290f4cf992045',
    'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species':
        '06615c1660c892fc199840c18123f8385b3542a8',
}


def _unsupported_pruning(*args, **kwargs):
    raise NotImplementedError('Attention head pruning is not supported for this NTv2 adapter.')


def _no_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is not None:
        raise NotImplementedError('Only head_mask=None is supported for this NTv2 adapter.')
    return [None] * num_hidden_layers


def _legacy_init_weights(self):
    # Legacy NTv2 constructors call init_weights directly, before 5.x metadata
    # exists. post_init registers it, then calls this method again safely.
    if not hasattr(self, 'all_tied_weights_keys'):
        self.post_init()
    else:
        PreTrainedModel.init_weights(self)


def _remote_initialize_weights(self, module, is_remote_code=False):
    # A local composite wrapper makes Transformers 5.3 pass False even when
    # visiting custom-code children. NTv2 writes .data directly, so loaded
    # tensors must be protected using the remote-code branch of this helper.
    PreTrainedModel._initialize_weights(self, module, is_remote_code=True)


def custom_nt_classes(source, revision=None, local_files_only=False):
    """Load pinned NTv2 code, adapting legacy methods only on its own classes."""
    added_stub = not hasattr(_pt_utils, 'find_pruneable_heads_and_indices')
    if added_stub:
        _pt_utils.find_pruneable_heads_and_indices = _unsupported_pruning
    try:
        cls = get_class_from_dynamic_module(
            'modeling_esm.EsmForMaskedLM', source, revision=revision,
            local_files_only=local_files_only,
        )
    finally:
        if added_stub:
            del _pt_utils.find_pruneable_heads_and_indices
    module = importlib.import_module(cls.__module__)
    if not hasattr(PreTrainedModel, 'get_head_mask'):
        module.EsmPreTrainedModel.get_head_mask = _no_head_mask
    if hasattr(PreTrainedModel, 'get_expanded_tied_weights_keys'):
        module.EsmPreTrainedModel.init_weights = _legacy_init_weights
    if 'is_remote_code' in inspect.signature(PreTrainedModel._initialize_weights).parameters:
        module.EsmPreTrainedModel._initialize_weights = _remote_initialize_weights
    return cls, module.EsmModel


def _normalize_backbone_config_fields(cfg):
    for attr, default in dict(is_decoder=False, add_cross_attention=False,
                              is_encoder_decoder=False, cross_attention_hidden_size=None).items():
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)
    return cfg


class SequenceClassificationConfig(PretrainedConfig):
    """Serializable encoder architecture, source revision and representation policy."""
    model_type = 'baculo-sequence-classification'

    def __init__(self, backbone_name_or_path: Optional[str] = None,
                 backbone_config: Optional[dict[str, Any]] = None,
                 backbone_revision: Optional[str] = None,
                 backbone_has_pooler: Optional[bool] = None,
                 hidden_dim: int = 512, num_labels: int = 2,
                 pooling: str = 'legacy', freeze_backbone: bool = True,
                 problem_type: str = 'single_label_classification', **kwargs):
        super().__init__(num_labels=num_labels, problem_type=problem_type, **kwargs)
        if pooling not in ('cls', 'legacy'):
            raise ValueError('pooling must be cls or legacy.')
        self.backbone_name_or_path = backbone_name_or_path
        self.backbone_config = backbone_config
        self.backbone_revision = backbone_revision
        self.backbone_has_pooler = backbone_has_pooler
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone


class SequenceClassification(PreTrainedModel):
    """Sequence classifier; frozen encoders always run in evaluation mode."""
    config_class = SequenceClassificationConfig
    base_model_prefix = 'backbone'
    main_input_name = 'input_ids'
    accepts_loss_kwargs = False

    def __init__(self, config, backbone=None):
        super().__init__(config)
        if config.backbone_config is None:
            raise ValueError('backbone_config is required.')
        if backbone is None:
            auto_map = config.backbone_config.get('auto_map', {}) or {}
            custom = 'AutoModelForMaskedLM' in auto_map and 'AutoModel' not in auto_map
            has_pooler = config.backbone_has_pooler
            if has_pooler is None:
                has_pooler = not custom  # Preserve old checkpoint representation.
            if custom:
                if not config.backbone_name_or_path:
                    raise ValueError('NTv2 reload requires backbone_name_or_path.')
                mlm_cls, encoder_cls = custom_nt_classes(
                    config.backbone_name_or_path, config.backbone_revision)
                backbone_cfg = mlm_cls.config_class.from_dict(config.backbone_config)
                _normalize_backbone_config_fields(backbone_cfg)
                backbone = encoder_cls(backbone_cfg, add_pooling_layer=has_pooler)
            else:
                config_cls = CONFIG_MAPPING[config.backbone_config['model_type']]
                backbone_cfg = config_cls.from_dict(config.backbone_config)
                _normalize_backbone_config_fields(backbone_cfg)
                backbone = AutoModel.from_config(backbone_cfg, add_pooling_layer=has_pooler)
        self.backbone = backbone
        self._backbone_forward_params = set(inspect.signature(backbone.forward).parameters)
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(nn.Linear(backbone.config.hidden_size, config.hidden_dim),
                                        nn.GELU(), nn.Linear(config.hidden_dim, config.num_labels))
        self.post_init()
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        values = dict(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids, **kwargs)
        values = {key: value for key, value in values.items()
                  if key in self._backbone_forward_params and value is not None}
        outputs = self.backbone(**values)
        if self.config.pooling == 'legacy' and getattr(outputs, 'pooler_output', None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=getattr(outputs, 'hidden_states', None),
                                        attentions=getattr(outputs, 'attentions', None))


def load_pretrained_nt(model_name, hidden_dim=512, model_path=None, revision=None,
                       local_files_only=False, pooling='cls', **kwargs):
    """Load NT with pinned code and an unpooled CLS representation for new probes.

    Old checkpoints retain their recorded/legacy pooling. Checkpoints depend on
    the original pinned custom code being cached or reachable on Hugging Face.
    """
    load_args = dict(trust_remote_code=True, local_files_only=local_files_only)
    if model_path:
        model = SequenceClassification.from_pretrained(model_path, **load_args)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **load_args)
        except OSError:
            # Old course checkpoints may not include tokenizer files.
            tokenizer = AutoTokenizer.from_pretrained(
                model.config.backbone_name_or_path,
                revision=model.config.backbone_revision, **load_args)
    else:
        revision = revision or DEFAULT_REVISIONS.get(model_name)
        pretrained_cfg = AutoConfig.from_pretrained(model_name, revision=revision, **load_args)
        _normalize_backbone_config_fields(pretrained_cfg)
        resolved_revision = getattr(pretrained_cfg, '_commit_hash', None) or revision
        auto_map = getattr(pretrained_cfg, 'auto_map', {}) or {}
        if 'AutoModelForMaskedLM' in auto_map and 'AutoModel' not in auto_map:
            mlm_cls, _ = custom_nt_classes(model_name, resolved_revision, local_files_only)
            mlm = mlm_cls.from_pretrained(model_name, config=pretrained_cfg,
                                         revision=resolved_revision, local_files_only=local_files_only)
            pretrained_backbone = mlm.esm
        else:
            pretrained_backbone = AutoModel.from_pretrained(
                model_name, config=pretrained_cfg, revision=resolved_revision,
                add_pooling_layer=(pooling == 'legacy'), **load_args)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision, **load_args)
        config = SequenceClassificationConfig(
            backbone_name_or_path=model_name,
            backbone_config=pretrained_backbone.config.to_dict(),
            backbone_revision=resolved_revision,
            backbone_has_pooler=getattr(pretrained_backbone, 'pooler', None) is not None,
            hidden_dim=hidden_dim, pooling=pooling, **kwargs)
        # Reuse the loaded encoder; do not allocate another full-sized backbone.
        model = SequenceClassification(config, backbone=pretrained_backbone)
    for attr in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            setattr(model.config, attr, value)
    return tokenizer, model
