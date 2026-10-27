"""Small CPU tests; custom-code checks require the pinned NTv2 code in cache."""
import torch
import pytest
import json
from transformers import AutoConfig, PreTrainedModel
from transformers.models.esm.configuration_esm import EsmConfig
from baculo_tss_probe.models import (
    SequenceClassification, SequenceClassificationConfig, custom_nt_classes,
)

NAME = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
REVISION = '06615c1660c892fc199840c18123f8385b3542a8'


def tiny_model(custom=False, pooling='cls'):
    if custom:
        cfg = AutoConfig.from_pretrained(NAME, revision=REVISION,
                                        trust_remote_code=True, local_files_only=True)
    else:
        cfg = EsmConfig(vocab_size=32, pad_token_id=1, mask_token_id=2,
                        token_dropout=False, position_embedding_type='rotary')
    cfg.hidden_size = 16
    cfg.intermediate_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.max_position_embeddings = 64
    cfg.vocab_size = 32
    cfg.hidden_dropout_prob = cfg.attention_probs_dropout_prob = 0.3
    return SequenceClassification(SequenceClassificationConfig(
        backbone_name_or_path=NAME if custom else None,
        backbone_revision=REVISION if custom else None,
        backbone_config=cfg.to_dict(), backbone_has_pooler=(pooling == 'legacy'),
        hidden_dim=8, pooling=pooling))


@pytest.mark.parametrize('custom', [False, True])
def test_roundtrip_preserves_all_weights_logits_and_freezing(tmp_path, custom):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = tiny_model(custom)
    model.train()
    assert model.training and model.classifier.training and not model.backbone.training
    ids = torch.tensor([[0, 5, 6, 7, 2], [0, 8, 9, 1, 1]])
    mask = ids.ne(1).long()
    expected = model(input_ids=ids, attention_mask=mask).logits.detach()
    assert torch.equal(expected, model(input_ids=ids, attention_mask=mask).logits)
    before = {key: value.clone() for key, value in model.backbone.state_dict().items()}
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
    loss = model(input_ids=ids, attention_mask=mask, labels=torch.tensor([0, 1])).loss
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    optimizer.step()
    assert all(torch.equal(value, model.backbone.state_dict()[key]) for key, value in before.items())
    model.eval()
    expected = model(input_ids=ids, attention_mask=mask).logits.detach()
    model.save_pretrained(tmp_path)
    restored = SequenceClassification.from_pretrained(tmp_path).eval()
    assert set(model.state_dict()) == set(restored.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(expected, restored(input_ids=ids, attention_mask=mask).logits,
                               rtol=0, atol=0)


def test_scoped_adapter_supports_mlm_construction_without_global_changes():
    original_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized
    original_has_head_mask = hasattr(PreTrainedModel, 'get_head_mask')
    mlm_class, _ = custom_nt_classes(NAME, REVISION, local_files_only=True)
    cfg = tiny_model(custom=True).backbone.config
    model = mlm_class(cfg).eval()
    assert hasattr(model, 'all_tied_weights_keys')
    assert PreTrainedModel.mark_tied_weights_as_initialized is original_mark_tied
    assert hasattr(PreTrainedModel, 'get_head_mask') == original_has_head_mask


def test_legacy_pooler_is_preserved_for_old_checkpoint_configs(tmp_path):
    model = tiny_model(pooling='legacy').eval()
    # The old config has no explicit policy or pooler flag.
    model.config.backbone_has_pooler = None
    assert model.backbone.pooler is not None
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path/'config.json').read_text())
    for field in ('pooling', 'backbone_has_pooler', 'backbone_revision', 'freeze_backbone'):
        saved.pop(field, None)
    (tmp_path/'config.json').write_text(json.dumps(saved))
    restored = SequenceClassification.from_pretrained(tmp_path).eval()
    assert restored.backbone.pooler is not None
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
