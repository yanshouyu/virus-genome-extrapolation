"""Reproduce a legacy NTv1 position-budget failure without loading model weights."""
import argparse
import json
from pathlib import Path

from Bio import SeqIO
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import EsmEmbeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--genbank', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sequence = str(SeqIO.read(args.genbank, 'genbank').seq[:6000]).upper()
    # Reproduce the legacy corpus settings, but allocate only a tiny embedding.
    encoded = tokenizer(sequence, max_length=1026, padding='max_length', truncation=True,
                        return_tensors='pt')
    padding_idx = config.pad_token_id
    mask = encoded['input_ids'].ne(padding_idx).int()
    positions = torch.cumsum(mask, dim=1) * mask + padding_idx
    tiny = EsmConfig(vocab_size=config.vocab_size, hidden_size=8,
                     max_position_embeddings=config.max_position_embeddings,
                     pad_token_id=padding_idx, mask_token_id=config.mask_token_id,
                     position_embedding_type=config.position_embedding_type,
                     token_dropout=config.token_dropout)
    embeddings = EsmEmbeddings(tiny).eval()
    try:
        with torch.no_grad():
            embeddings(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
    except IndexError as error:
        failure = str(error)
    else:
        raise AssertionError('Expected legacy positional lookup to fail.')
    safe_budget = config.max_position_embeddings - padding_idx - 1
    corrected = tokenizer(sequence, max_length=safe_budget, truncation=True,
                          padding='max_length', return_tensors='pt')
    with torch.no_grad():
        actual = embeddings(input_ids=corrected['input_ids'], attention_mask=corrected['attention_mask'])
    assert torch.isfinite(actual).all()
    result = dict(model_dir=args.model_dir, input_nt=len(sequence),
                  position_embedding_type=config.position_embedding_type,
                  position_embedding_rows=config.max_position_embeddings,
                  tokenizer_model_max_length=tokenizer.model_max_length,
                  legacy_padded_tokens=encoded['input_ids'].shape[1],
                  legacy_nonpad_tokens=int(mask.sum()), max_position_id=int(positions.max()),
                  legacy_error=failure, safe_nonpad_token_budget=safe_budget,
                  corrected_shape=list(actual.shape), corrected_forward_finite=True,
                  scope='CPU positional embeddings only; no pretrained weights or adaptation training')
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
