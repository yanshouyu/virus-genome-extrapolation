"""Fixed duration-study schedule and explicit low-rank adapter layers."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def learning_rate_scale(step, total_steps, warmup=26):
    if total_steps <= warmup or not 1 <= step <= total_steps:
        raise ValueError('Invalid schedule bounds')
    if step <= warmup:
        return step / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))


def plateau_summary(history, horizon=1024, relative_tolerance=.005):
    """Descriptive saturation check, never a downstream-driven stopping rule."""
    end = history[-1]['step']
    earlier = [h['mlm_loss'] for h in history if h['step'] <= end - horizon]
    if not earlier:
        return dict(assessable=False)
    old, new = min(earlier), min(h['mlm_loss'] for h in history)
    improvement = (old - new) / old
    return dict(assessable=True, horizon=horizon, relative_tolerance=relative_tolerance,
                best_loss_relative_improvement=improvement,
                plateau_by_rule=improvement < relative_tolerance)


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x; frozen base, zero initial update, no dropout."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('rank and alpha must be positive')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * (self.alpha / self.rank)


def inject_lora(model, rank=8, alpha=16):
    model.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules()
               if name.startswith('esm.encoder.layer.')
               and name.endswith(('.attention.self.query', '.attention.self.value'))
               and isinstance(module, nn.Linear)]
    if len(targets) != 2 * model.config.num_hidden_layers:
        raise ValueError(f'Expected query/value in each layer, found {len(targets)}')
    for name, module in targets:
        parent, key = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), key, LoRALinear(module, rank, alpha))
    return [name for name, _ in targets]


def state_to_save(model, method):
    if method == 'full':
        return model.state_dict()
    return {name: parameter.detach() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def load_training_state(model, state, method):
    if method == 'full':
        model.load_state_dict(state, strict=True)
    else:
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError('Adapter checkpoint keys do not match trainable parameters')
        result = model.load_state_dict(state, strict=False)
        assert not result.unexpected_keys
    assert all(torch.equal(model.state_dict()[key].detach().cpu(), value.cpu())
               for key, value in state.items())
