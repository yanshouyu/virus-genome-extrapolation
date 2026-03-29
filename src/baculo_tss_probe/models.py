import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Any, Optional, cast

# not all NT models are to be experimented
NT_MODELS = [
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-500m-1000g",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
]


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
        self.backbone = AutoModel.from_config(backbone_config)
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
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

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
    assert model_name in NT_MODELS

    base_source = model_path if model_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(base_source)

    if model_path:
        model = SequenceClassification.from_pretrained(model_path)
    else:
        pretrained_backbone = AutoModel.from_pretrained(model_name)
        config = SequenceClassificationConfig(
            backbone_name_or_path=model_name,
            backbone_config=pretrained_backbone.config.to_dict(),
            hidden_dim=hidden_dim,
            **kwargs,
        )
        model = SequenceClassification(config)
        model.backbone = pretrained_backbone
        model.config.backbone_name_or_path = model_name
    
    for _, param in model.backbone.named_parameters():
        param.requires_grad = False

    return tokenizer, model


