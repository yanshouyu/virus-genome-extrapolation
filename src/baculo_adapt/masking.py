"""Explicit masking and token-budget checks for controlled MLM training."""
import torch


def mask_tokens(ids,tokenizer,generator):
    """BERT 15% masking; no labels or random replacements on special IDs."""
    special=torch.isin(ids,torch.tensor(tokenizer.all_special_ids,dtype=ids.dtype))
    eligible=~special
    chosen=(torch.rand(ids.shape,generator=generator)<.15)&eligible
    for i in range(len(ids)):
        if not chosen[i].any():
            valid=torch.where(eligible[i])[0]
            if not len(valid):raise ValueError('No ordinary tokens in sample')
            chosen[i,valid[torch.randint(len(valid),(1,),generator=generator)]]=True
    labels=ids.clone();labels[~chosen]=-100
    result=ids.clone();action=torch.rand(ids.shape,generator=generator)
    result[chosen&(action<.8)]=tokenizer.mask_token_id
    vocabulary=torch.tensor([i for i in range(len(tokenizer)) if i not in tokenizer.all_special_ids])
    replacement=vocabulary[torch.randint(len(vocabulary),ids.shape,generator=generator)]
    random_sites=chosen&(action>=.8)&(action<.9)
    result[random_sites]=replacement[random_sites]
    assert torch.all(labels[special]==-100)
    return result,labels


def check_token_budget(ids,attention,config,tokenizer):
    """Validate emitted non-padding lengths against actual model/tokenizer limits."""
    limits=[int(config.max_position_embeddings)]
    if tokenizer.model_max_length<10**8:limits.append(int(tokenizer.model_max_length))
    if getattr(config,'position_embedding_type','rotary')=='absolute':
        limits.append(int(config.max_position_embeddings)-int(config.pad_token_id)-1)
    limit=min(limits)
    if int(attention.sum(1).max())>limit:raise ValueError('Token budget exceeded')
    assert ids.shape==attention.shape
    return limit
