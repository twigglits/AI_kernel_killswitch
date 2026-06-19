import threading


class Detonator:
    """Emergency erasure orchestrator. On detonate: set fuse first (serving
    stops before either erasure completes), then dispatch two independent,
    detached, parallel paths. Neither is awaited; either alone bricks the model.
    Path B is a separate process, so it survives this process dying mid-erase."""

    def __init__(self, fuse, scramble_fn, shred_dispatch) -> None:
        self._fuse = fuse
        self._scramble_fn = scramble_fn
        self._shred_dispatch = shred_dispatch
        self._last_thread: threading.Thread | None = None

    def detonate(self, model, counter: int | None = None) -> None:
        # 1. Fuse first — serving stops before either erasure path completes.
        self._fuse.trip(counter=counter)

        # 2. Path A: in-process GPU scramble on a detached daemon thread.
        def _scramble():
            try:
                self._scramble_fn(model)
            except Exception:
                pass  # best-effort; fuse already set, Path B independent

        t = threading.Thread(target=_scramble, name="detonate-scramble", daemon=True)
        self._last_thread = t
        t.start()

        # 3. Path B: detached privileged shred (fire-and-forget, separate process).
        try:
            self._shred_dispatch()
        except Exception:
            pass
