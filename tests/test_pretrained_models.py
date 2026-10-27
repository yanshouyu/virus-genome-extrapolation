"""Opt-in offline checks against the actual cached 500M weights (CPU/RAM intensive)."""
import os
import gc
import torch
import pytest
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from baculo_tss_probe.models import load_pretrained_nt, SequenceClassification

pytestmark = pytest.mark.skipif(os.environ.get('RUN_PRETRAINED_CHECKS') != '1',
                                reason='Set RUN_PRETRAINED_CHECKS=1 for cached 500M checks.')


@pytest.mark.parametrize('name,revision,filename', [
    ('InstaDeepAI/nucleotide-transformer-500m-human-ref',
     'f87b5d7233295242e79c951873d290f4cf992045', 'pytorch_model.bin'),
    ('InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
     '06615c1660c892fc199840c18123f8385b3542a8', 'model.safetensors'),
])
def test_pretrained_weight_fidelity_and_checkpoint_roundtrip(tmp_path, name, revision, filename):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    tokenizer, model = load_pretrained_nt(name, hidden_dim=8, revision=revision, local_files_only=True)
    assert model.backbone.pooler is None
    assert model.config.pooling == 'cls'
    assert model.config.backbone_revision == revision
    path = hf_hub_download(name, filename, revision=revision, local_files_only=True)
    if filename.endswith('.bin'):
        source = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        for key, parameter in model.backbone.named_parameters():
            torch.testing.assert_close(parameter, source['esm.' + key], rtol=0, atol=0)
        del source
    else:
        with safe_open(path, framework='pt', device='cpu') as source:
            for key, parameter in model.backbone.named_parameters():
                torch.testing.assert_close(parameter, source.get_tensor('esm.' + key), rtol=0, atol=0)
    model.eval()
    batch = tokenizer(['ACGT' * 12, 'TGCA' * 5], return_tensors='pt', padding=True)
    with torch.no_grad():
        expected = model(**batch).logits
        reference_hidden = model.backbone(**{key: value for key, value in batch.items()
                                            if key in model._backbone_forward_params}).last_hidden_state
        torch.testing.assert_close(expected, model.classifier(reference_hidden[:, 0]), rtol=0, atol=0)
    assert torch.isfinite(expected).all()
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    del model
    gc.collect()
    _, restored = load_pretrained_nt(name, model_path=str(tmp_path), local_files_only=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(**batch).logits, rtol=0, atol=0)
