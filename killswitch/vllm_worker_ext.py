from killswitch.scramble import scramble_parameters


class KillswitchWorkerExtension:
    """Mixed into the vLLM worker via ``worker_extension_cls``.

    Runs INSIDE the worker (EngineCore) process, where the model's GPU tensors
    live, so the in-place scramble reaches the real weights even though vLLM V1
    runs the model in a separate process. Invoked by name through
    ``collective_rpc("scramble_weights")`` — a plain string method name, so no
    insecure function serialization is required.
    """

    def scramble_weights(self) -> int:
        # self.model_runner is provided by the vLLM GPU worker this is mixed into.
        return scramble_parameters(self.model_runner.model)
