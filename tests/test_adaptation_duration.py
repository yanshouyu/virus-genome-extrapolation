import pytest
import torch
from torch import nn
from baculo_adapt.duration import (LoRALinear, learning_rate_scale, plateau_summary,
                                   inject_lora, state_to_save, load_training_state)


def test_schedule_warmup_end_and_invalid_bounds():
    assert learning_rate_scale(26, 4096) == 1
    assert learning_rate_scale(4096, 4096) == 0
    assert 0 < learning_rate_scale(1024, 4096) < 1
    with pytest.raises(ValueError):
        learning_rate_scale(1, 5)


def test_plateau_uses_prior_best_and_requires_full_horizon():
    assert not plateau_summary([dict(step=512, mlm_loss=5)])['assessable']
    assert plateau_summary([dict(step=3072, mlm_loss=5), dict(step=4096, mlm_loss=4.99)])['plateau_by_rule']
    assert not plateau_summary([dict(step=3072, mlm_loss=5), dict(step=4096, mlm_loss=4.9)])['plateau_by_rule']


def test_lora_zero_start_frozen_base_and_exact_reload():
    torch.manual_seed(42)
    base = nn.Linear(5, 7)
    x = torch.randn(3, 5)
    expected = base(x).detach()
    layer = LoRALinear(base, 2, 4)
    assert torch.equal(layer(x), expected)
    initial = {key: value.clone() for key, value in base.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.01)
    layer(x).square().mean().backward()
    optimizer.step()
    assert torch.count_nonzero(layer.lora_B) > 0
    assert all(torch.equal(initial[k], v) for k, v in base.state_dict().items())
    state = state_to_save(layer, 'lora')
    restored = LoRALinear(nn.Linear(5, 7), 2, 4)
    restored.base.load_state_dict(initial)
    load_training_state(restored, state, 'lora')
    assert torch.equal(layer(x), restored(x))
    with pytest.raises(ValueError):
        load_training_state(restored, {}, 'lora')


def test_injected_custom_nt_backward_matches_zero_adapter_baseline():
    from transformers import AutoConfig
    from baculo_tss_probe.models import custom_nt_classes, _normalize_backbone_config_fields
    name = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
    rev = '06615c1660c892fc199840c18123f8385b3542a8'
    cfg = AutoConfig.from_pretrained(name, revision=rev, trust_remote_code=True, local_files_only=True)
    cfg.hidden_size=16; cfg.intermediate_size=32; cfg.num_hidden_layers=1
    cfg.num_attention_heads=2; cfg.vocab_size=32; cfg.max_position_embeddings=64
    _normalize_backbone_config_fields(cfg)
    cls, _ = custom_nt_classes(name, rev, True)
    model = cls(cfg).eval()
    ids = torch.tensor([[0,4,5,6,7]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    targets = inject_lora(model, 2, 4)
    assert len(targets) == 2
    with torch.no_grad():
        assert torch.equal(expected, model(input_ids=ids).logits)
    before = {k: v.clone() for k,v in model.named_parameters() if not v.requires_grad}
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    assert any(torch.count_nonzero(v) for k,v in model.named_parameters() if k.endswith('lora_B'))
    assert all(torch.equal(before[k],v) for k,v in model.named_parameters() if not v.requires_grad)
