import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

# not all NT models are to be experimented
NT_MODELS = [
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-500m-1000g",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
]


def load_pretrained_nt(model_name, model_path: Optional[str] = None, **kwargs):
    "load pretrained nucleotide transformer model by name"
    assert model_name in NT_MODELS
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TODO: make the classification head adjustable by customize the model 
    if model_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=2
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
    model.config.problem_type = 'single_label_classification'
    
    for name, param in model.named_parameters():
        if name.startswith("esm"):
            param.requires_grad = False
    
    return tokenizer, model


