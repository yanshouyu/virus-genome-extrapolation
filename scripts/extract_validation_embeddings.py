"""Validation-only frozen embeddings. Deliberately never opens test.csv."""
import argparse, csv, hashlib, importlib.metadata, json, time
from pathlib import Path
import numpy as np
import torch
from baculo_tss_probe.models import load_pretrained_nt, DEFAULT_REVISIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['ntv1', 'ntv2'], required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    source = list(DEFAULT_REVISIONS)[0 if args.model == 'ntv1' else 1]
    started = time.time()
    torch.manual_seed(42)
    tokenizer, model = load_pretrained_nt(source, local_files_only=True)
    model = model.cuda().eval()
    metadata = dict(source=source, revision=model.config.backbone_revision,
                    packages={k: importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), splits={})
    for split in ['train','validation']:
        path = Path('data/processed/clean_v2') / (split+'.csv')
        rows = list(csv.DictReader(path.open()))
        collected = {'cls': [], 'mean': []}; replay = None
        for start in range(0, len(rows), 32):
            texts = [r['sequence'] for r in rows[start:start+32]]
            tokens = tokenizer(texts, padding=True, truncation=False,
                               return_tensors='pt')
            special = torch.isin(tokens['input_ids'], torch.tensor(tokenizer.all_special_ids)).cuda()
            inputs = {k:v.cuda() for k,v in tokens.items() if k in model._backbone_forward_params}
            with torch.inference_mode():
                hidden = model.backbone(**inputs).last_hidden_state
                mask = inputs['attention_mask'].bool() & ~special
                assert torch.all(mask.sum(1)>0)
                pooled = (hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                if start == 0:
                    again = model.backbone(**inputs).last_hidden_state
                    replay = float((hidden-again).abs().max().cpu())
                    assert replay == 0, replay
            collected['cls'].append(hidden[:,0].cpu().numpy())
            collected['mean'].append(pooled.cpu().numpy())
        arrays = {k:np.concatenate(v) for k,v in collected.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(out/(split+'.npz'), **arrays,
                            labels=np.array([int(r['label']) for r in rows]),
                            names=np.array([r['name'] for r in rows]),
                            groups=np.array([r['group'] for r in rows]))
        metadata['splits'][split] = dict(n=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        replay_max_abs=replay, dimensions=arrays['cls'].shape[1])
    metadata['elapsed_seconds'] = time.time()-started
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    main()

