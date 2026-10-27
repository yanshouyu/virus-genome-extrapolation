# Audit evidence, 2026-10-13

Baseline commit: `42b39b59462238bca11a6a1563b212c3e33c9009` (`dev`).

- `dataset.json`: output of `python scripts/audit_dataset.py`, including input SHA-256 hashes. Exit 0; it reports findings rather than enforcing a split policy.
- `checkpoint-check.txt`: output of `PYTHONPATH=src python scripts/check_nt_checkpoint.py`. **Expected observed exit: 1**, because the existing NTv2 reload is broken. It is a diagnostic failure, not a passing test. Uses tiny randomly initialized weights and cached custom code, not trained weights.
- `model-check.txt`: offline CPU smoke check of the two cached 500M pretrained models. Both returned finite logits. The newly initialized classifier is random, so the numeric logits themselves are not reference predictions. NTv1 reports missing pooler weights.

All model checks used WSL Ubuntu's existing `ece657d` Python, CPU only and two computation threads. No network model downloads, training or environment modifications. The main assessment records versions and cached model revisions. Tiny checkpoint testing does not establish correctness of trained historical checkpoints.

The model smoke check used this core loop after importing the project's existing loader (which installs its compatibility patches):

```python
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from baculo_tss_probe.models import load_pretrained_nt
for name in [
    'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
    'InstaDeepAI/nucleotide-transformer-500m-human-ref',
]:
    tokenizer, model = load_pretrained_nt(name, hidden_dim=8)
    model.eval()
    batch = tokenizer(['ACGT' * 12, 'TGCA' * 12], return_tensors='pt', padding=True)
    with torch.no_grad():
        logits = model(**batch).logits
    assert torch.isfinite(logits).all()
    del model, tokenizer
```

Source documents were the user's February 19 proposal and April 2 final report, read from their provided OneDrive paths; originals were not copied into the repository or modified. Saved histories remain in the original local, ignored `notebooks/trainer_states/` directory. The assessment distinguishes last-step values from minima selected using evaluation data.
