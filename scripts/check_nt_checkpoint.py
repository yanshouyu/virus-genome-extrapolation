"""Reproduce NTv2 checkpoint reconstruction on CPU with tiny random weights.

Requires the NTv2 config/code already cached; never downloads weights or trains.
Run from the repository root with the existing ece657d environment:
    PYTHONPATH=src python scripts/check_nt_checkpoint.py
A nonzero exit indicates the checkpoint round trip failed.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import importlib
import tempfile
import torch
from transformers import AutoConfig
from baculo_tss_probe.models import SequenceClassification, SequenceClassificationConfig, custom_nt_classes


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    name = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
    revision = '06615c1660c892fc199840c18123f8385b3542a8'
    cfg = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True,
                                     local_files_only=True)
    mlm_class, _ = custom_nt_classes(name, revision, local_files_only=True)
    remote = importlib.import_module(mlm_class.__module__)
    cfg.hidden_size = 16
    cfg.intermediate_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.vocab_size = 32
    cfg.max_position_embeddings = 64
    for key, value in dict(is_decoder=False, add_cross_attention=False,
                           is_encoder_decoder=False, cross_attention_hidden_size=None).items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    encoder = remote.EsmModel(cfg, add_pooling_layer=False)
    wrapper = SequenceClassification(SequenceClassificationConfig(
        backbone_name_or_path=name, backbone_revision=revision,
        backbone_config=cfg.to_dict(), backbone_has_pooler=False,
        pooling='cls', hidden_dim=8))
    print('Original encoder:', type(encoder))
    print('Reconstructed encoder:', type(wrapper.backbone))
    print('Original intermediate weight:', tuple(encoder.encoder.layer[0].intermediate.dense.weight.shape))
    print('Reconstructed intermediate weight:', tuple(wrapper.backbone.encoder.layer[0].intermediate.dense.weight.shape))
    wrapper.backbone = encoder
    wrapper.eval()
    tokens = torch.tensor([[0, 5, 6, 7, 2]])
    with torch.no_grad():
        expected = wrapper(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits
    with tempfile.TemporaryDirectory() as directory:
        wrapper.save_pretrained(directory)
        restored = SequenceClassification.from_pretrained(directory).eval()
        with torch.no_grad():
            actual = restored(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits
        torch.testing.assert_close(actual, expected)
    print('Checkpoint round trip preserved logits.')


if __name__ == '__main__':
    main()
