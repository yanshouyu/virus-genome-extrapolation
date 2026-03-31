import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import transformers.pytorch_utils as _pt_utils
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Any, Optional, cast

# Some trust_remote_code model files (e.g. NT v2) import find_pruneable_heads_and_indices
# which was removed in transformers 5.x. Provide a stub so the import succeeds; the
# function is only called for attention head pruning which we never do.
if not hasattr(_pt_utils, "find_pruneable_heads_and_indices"):
    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        raise NotImplementedError("Head pruning is not supported with this version of transformers.")
    _pt_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

# NT v2's custom EsmForMaskedLM calls init_weights() (old transformers 4.x API) instead of
# post_init() (transformers 5.x API), so all_tied_weights_keys is never set on the model.
# Patch mark_tied_weights_as_initialized to lazily compute it when missing.
_orig_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized
def _patched_mark_tied(self, loading_info):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=True)
    _orig_mark_tied(self, loading_info)
PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied
# NT v2's custom EsmModel.forward calls get_head_mask which was removed from
# PreTrainedModel in transformers 5.x. head_mask is always None in our usage
# (no attention-head masking), so returning [None] * n is correct.
if not hasattr(PreTrainedModel, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("Non-None head_mask is not supported with this version of transformers.")
        return [None] * num_hidden_layers
    PreTrainedModel.get_head_mask = _get_head_mask





def _normalize_backbone_config_fields(cfg):
    """Populate common optional decoder fields expected by some model implementations."""
    defaults = {
        "is_decoder": False,
        "add_cross_attention": False,
        "is_encoder_decoder": False,
        "cross_attention_hidden_size": None,
    }
    for attr, default in defaults.items():
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)
    return cfg


class SequenceClassificationConfig(PretrainedConfig):
    """Configuration for a sequence classifier built on top of a pretrained backbone."""

    model_type = "baculo-sequence-classification"

    def __init__(
        self,
        backbone_name_or_path: Optional[str] = None,
        backbone_config: Optional[dict[str, Any]] = None,
        hidden_dim: int = 512,
        num_labels: int = 2,
        problem_type: str = "single_label_classification",
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, problem_type=problem_type, **kwargs)
        self.backbone_name_or_path = backbone_name_or_path
        self.backbone_config = backbone_config
        self.hidden_dim = hidden_dim


class SequenceClassification(PreTrainedModel):
    """Sequence classifier with a pretrained base model and an MLP head."""

    config_class = SequenceClassificationConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"

    def __init__(
        self,
        config: SequenceClassificationConfig,
    ):
        super().__init__(config)

        if config.backbone_config is None:
            raise ValueError("SequenceClassificationConfig must define backbone_config.")

        backbone_model_type = config.backbone_config.get("model_type")
        if backbone_model_type is None:
            raise ValueError("backbone_config must define model_type.")

        backbone_config_class = CONFIG_MAPPING[backbone_model_type]
        backbone_config = backbone_config_class.from_dict(config.backbone_config)
        backbone_config = _normalize_backbone_config_fields(backbone_config)
        self.backbone = AutoModel.from_config(backbone_config)
        self._backbone_forward_params = set(inspect.signature(self.backbone.forward).parameters)
        self.num_labels = config.num_labels

        input_dim = getattr(backbone_config, "hidden_size", None)
        if input_dim is None:
            raise ValueError("Base model config must define hidden_size.")

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_labels),
        )
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }
        if token_type_ids is not None and "token_type_ids" in self._backbone_forward_params:
            backbone_kwargs["token_type_ids"] = token_type_ids

        filtered_backbone_kwargs = {
            k: v
            for k, v in backbone_kwargs.items()
            if k in self._backbone_forward_params and v is not None
        }
        outputs = self.backbone(**filtered_backbone_kwargs)

        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            # Fallback for backbones without pooler: use first token embedding.
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return SequenceClassifierOutput(
            loss=cast(Optional[torch.FloatTensor], loss),
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def load_pretrained_nt(
    model_name,
    hidden_dim: int = 512,
    model_path: Optional[str] = None,
    **kwargs,
):
    "load pretrained nucleotide transformer model by name"
    base_source = model_path if model_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(base_source, trust_remote_code=True)

    if model_path:
        model = SequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    else:
        pretrained_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _normalize_backbone_config_fields(pretrained_cfg)

        auto_map = getattr(pretrained_cfg, "auto_map", {}) or {}
        if "AutoModelForMaskedLM" in auto_map and "AutoModel" not in auto_map:
            # Model has a custom architecture class registered only for MLM (e.g. NT v2
            # with SwiGLU FFN using 2*intermediate_size). Load MLM and strip the head.
            mlm = AutoModelForMaskedLM.from_pretrained(
                model_name, config=pretrained_cfg, trust_remote_code=True
            )
            pretrained_backbone = mlm.esm
        else:
            pretrained_backbone = AutoModel.from_pretrained(
                model_name, config=pretrained_cfg, trust_remote_code=True
            )

        config = SequenceClassificationConfig(
            backbone_name_or_path=model_name,
            backbone_config=pretrained_backbone.config.to_dict(),
            hidden_dim=hidden_dim,
            **kwargs,
        )
        model = SequenceClassification(config)
        model.backbone = pretrained_backbone
        model.config.backbone_name_or_path = model_name
    
    # Keep model and generation configs aligned with tokenizer special-token IDs.
    special_token_attrs = ("pad_token_id", "bos_token_id", "eos_token_id")
    for attr in special_token_attrs:
        tok_val = getattr(tokenizer, attr, None)
        if tok_val is not None:
            setattr(model.config, attr, tok_val)

    if getattr(model, "generation_config", None) is not None:
        for attr in special_token_attrs:
            tok_val = getattr(tokenizer, attr, None)
            if tok_val is not None:
                setattr(model.generation_config, attr, tok_val)

    # freeze base model weights
    for _, param in model.backbone.named_parameters():
        param.requires_grad = False

    return tokenizer, model


