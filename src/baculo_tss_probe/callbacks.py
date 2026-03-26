import os, glob
from transformers import TrainerCallback
from safetensors.torch import save_file


class SaveHeadCallback(TrainerCallback):
    def __init__(self, prune_full_model: bool = False):
        self.prune_full_model = prune_full_model

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        sd = model.state_dict()
        head_sd = {k: v.cpu() for k, v in sd.items() if k.startswith("classifier.")}
        if head_sd:
            save_file(head_sd, os.path.join(output_dir, "head.safetensors"))

        if self.prune_full_model:
            # Remove full model shards to save space (disables resume-from-checkpoint)
            patterns = [
                "pytorch_model.bin", "pytorch_model*.bin", "pytorch_model.bin.index.json",
                "model.safetensors", "model-*.safetensors", "model.safetensors.index.json"
            ]
            for pat in patterns:
                for path in glob.glob(os.path.join(output_dir, pat)):
                    try: os.remove(path)
                    except FileNotFoundError: pass
