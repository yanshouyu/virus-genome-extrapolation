"""Check a saved probe reproduces its recorded test logits on a GPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from baculo_tss_probe.data import prep_ds
from baculo_tss_probe.models import load_pretrained_nt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    directory = Path(args.run_dir)
    cfg = json.loads((directory/'config.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('This verification requires CUDA.')
    tokenizer, model = load_pretrained_nt(cfg['base_model'], model_path=str(directory/'final-model'))
    model.eval().to('cuda')
    dataset = prep_ds('virus', tokenizer, data_dir=cfg['data_dir'], seed=cfg['data_seed'])['test']
    expected = np.load(directory/'test_predictions.npz')
    logits, labels = [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), cfg['per_device_eval_batch_size']):
            records = [dataset[i] for i in range(start, min(start+cfg['per_device_eval_batch_size'], len(dataset)))]
            labels.extend(int(row['labels']) for row in records)
            encoded = tokenizer.pad([{k: v for k, v in row.items() if k != 'labels'} for row in records],
                                    return_tensors='pt').to('cuda')
            logits.append(model(**encoded).logits.cpu().numpy())
    actual = np.concatenate(logits)
    np.testing.assert_array_equal(labels, expected['labels'])
    np.testing.assert_allclose(actual, expected['logits'], rtol=1e-5, atol=1e-5)
    properties = torch.cuda.get_device_properties(0)
    result = dict(run=str(directory), samples=len(labels), cuda_device=properties.name,
                  device_memory_bytes=properties.total_memory, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda,
                  max_absolute_logit_difference=float(np.max(np.abs(actual-expected['logits']))),
                  tolerance=dict(rtol=1e-5, atol=1e-5), passed=True)
    (directory/'gpu_reload_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
