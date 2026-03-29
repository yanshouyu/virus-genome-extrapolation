import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional

# not all NT models are to be experimented
NT_MODELS = [
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-500m-1000g",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
]


class SequenceClassification(nn.Module):
    """Sequence classifier with a pretrained base model and an MLP head."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        num_labels: int = 2,
        model_path: Optional[str] = None,
    ):
        super().__init__()

        base_source = model_path if model_path else model_name
        self.base_model = AutoModel.from_pretrained(base_source)
        self.config = self.base_model.config
        self.num_labels = num_labels
        self.config.num_labels = num_labels
        self.config.problem_type = "single_label_classification"

        input_dim = getattr(self.config, "hidden_size", None)
        if input_dim is None:
            raise ValueError("Base model config must define hidden_size.")

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.base_model(
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
            loss=loss,
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = SequenceClassification(
        model_name=model_name,
        model_path=model_path,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    
    for name, param in model.base_model.named_parameters():
        if name.startswith("esm"):
            param.requires_grad = False
    
    return tokenizer, model


