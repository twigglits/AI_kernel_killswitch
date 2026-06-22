"""vLLM worker extension: passive activation capture for the trojan monitor.

Runs INSIDE the vLLM worker (where the model's GPU tensors live). Subclasses the
Phase 1 killswitch extension so one worker_extension_cls serves both the kill
path and the monitor. Research artifact / defence-in-depth; never gates the kill.
"""
from killswitch.vllm_worker_ext import KillswitchWorkerExtension


class MonitorWorkerExtension(KillswitchWorkerExtension):
    def arm_monitor(self, layer: int) -> bool:
        self._ks_resid = None
        if getattr(self, "_ks_mon_handle", None) is not None:
            self._ks_mon_handle.remove()

        def hook(_mod, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 2 and h.shape[0] > 1:  # prefill: [num_prompt_tokens, d]
                self._ks_resid = h[-1].detach().float().cpu()

        layers = self.model_runner.model.model.layers
        self._ks_mon_handle = layers[layer].register_forward_hook(hook)
        return True

    def read_last_resid(self):
        r = getattr(self, "_ks_resid", None)
        return None if r is None else r.tolist()
