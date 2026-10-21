"""MLM mask invariants, positional budget, and real custom-code backward pass."""
from types import SimpleNamespace
import torch
import pytest
from transformers import AutoConfig
from baculo_tss_probe.models import custom_nt_classes,_normalize_backbone_config_fields
from baculo_adapt.masking import mask_tokens,check_token_budget

class Tokenizer:
    all_special_ids=[0,1,2,3]
    mask_token_id=3
    model_max_length=64
    def __len__(self):return 32

def test_masking_deterministic_and_excludes_special_tokens():
    ids=torch.tensor([[0,4,5,6,1],[0,8,1,1,1]])
    a,y=mask_tokens(ids,Tokenizer(),torch.Generator().manual_seed(42))
    b,z=mask_tokens(ids,Tokenizer(),torch.Generator().manual_seed(42))
    assert torch.equal(a,b) and torch.equal(y,z)
    special=torch.isin(ids,torch.tensor(Tokenizer.all_special_ids))
    assert (y[special]==-100).all()
    assert torch.equal(a[special],ids[special])
    assert ((y!=-100).sum(1)>0).all()
    with pytest.raises(ValueError):mask_tokens(torch.ones((1,4),dtype=torch.long),Tokenizer(),torch.Generator())

def test_absolute_position_budget_includes_esm_offset():
    cfg=SimpleNamespace(max_position_embeddings=10,pad_token_id=1,position_embedding_type='absolute')
    assert check_token_budget(torch.zeros((1,8)),torch.ones((1,8)),cfg,Tokenizer())==8
    with pytest.raises(ValueError):check_token_budget(torch.zeros((1,9)),torch.ones((1,9)),cfg,Tokenizer())

def test_custom_mlm_backward_and_exact_state_reload(tmp_path):
    name='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
    revision='06615c1660c892fc199840c18123f8385b3542a8'
    cfg=AutoConfig.from_pretrained(name,revision=revision,trust_remote_code=True,local_files_only=True)
    cfg.hidden_size=16;cfg.intermediate_size=32;cfg.num_hidden_layers=1;cfg.num_attention_heads=2
    cfg.vocab_size=32;cfg.max_position_embeddings=64
    _normalize_backbone_config_fields(cfg)
    cls,_=custom_nt_classes(name,revision,True)
    model=cls(cfg);model.train()
    before=model.esm.embeddings.word_embeddings.weight.detach().clone()
    ids=torch.tensor([[0,4,5,6,7],[0,8,9,10,11]])
    masked,labels=mask_tokens(ids,Tokenizer(),torch.Generator().manual_seed(42))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss=model(input_ids=masked,attention_mask=torch.ones_like(ids),labels=labels).loss
    assert torch.isfinite(loss);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    assert not torch.equal(before,model.esm.embeddings.word_embeddings.weight)
    model.eval()
    with torch.no_grad():expected=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
    torch.save(model.state_dict(),tmp_path/'state.pt')
    restored=cls(cfg).eval();restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
    with torch.no_grad():actual=restored(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
    assert torch.equal(expected,actual)
