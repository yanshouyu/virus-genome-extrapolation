"""Compare saved probe backbones exactly against pinned cached pretrained weights."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    for family in ('ntv1', 'ntv2'):
        first = json.loads((root/f'{family}-seed42/config.json').read_text())
        snapshot = Path(snapshot_download(first['base_model'], revision=first['model_revision'],
                                          local_files_only=True))
        with ExitStack() as stack:
            if (snapshot/'model.safetensors').is_file():
                reference = stack.enter_context(safe_open(snapshot/'model.safetensors', framework='pt'))
                keys = set(reference.keys())
                get_tensor = reference.get_tensor
            else:
                reference = torch.load(snapshot/'pytorch_model.bin', map_location='cpu',
                                       weights_only=True, mmap=True)
                keys = set(reference)
                get_tensor = reference.__getitem__
            for seed in (42, 43, 44):
                directory = root/f'{family}-seed{seed}'
                cfg = json.loads((directory/'config.json').read_text())
                assert (cfg['base_model'], cfg['model_revision']) == (first['base_model'], first['model_revision'])
                checked, elements = 0, 0
                with safe_open(directory/'final-model/model.safetensors', framework='pt') as saved:
                    for key in saved.keys():
                        if not key.startswith('backbone.'):
                            continue
                        short = key.removeprefix('backbone.')
                        source_key = 'esm.'+short if 'esm.'+short in keys else short
                        if source_key not in keys:
                            raise AssertionError(f'Missing pretrained reference for {key}')
                        actual, expected = saved.get_tensor(key), get_tensor(source_key)
                        if not torch.equal(actual, expected):
                            raise AssertionError(f'Encoder tensor changed: {directory}/{key}')
                        checked += 1
                        elements += actual.numel()
                assert checked > 100
                result = dict(run=str(directory), model=cfg['base_model'], revision=cfg['model_revision'],
                              reference_snapshot=str(snapshot), tensors_checked=checked,
                              elements_checked=elements, equality='exact torch.equal', passed=True)
                (directory/'frozen_encoder_check.json').write_text(json.dumps(result, indent=2)+'\n')
                print(json.dumps(result), flush=True)
        del get_tensor, reference


if __name__ == '__main__':
    main()
