"""Compatibility shim for loading this project's own Lightning checkpoints.

PyTorch 2.6 changed ``torch.load`` to default ``weights_only=True``, which
refuses to unpickle anything but tensors and a small allowlist of builtins.
Lightning checkpoints written here embed the Hydra-instantiated hyperparameters
— sampler, aggregator, scaler and preprocessor objects — so loading one to run
``trainer.test(ckpt_path=...)`` raises ``UnpicklingError``.

The allowlist approach does not fit: the objects involved are open-ended and
change with every config, so enumerating them would need updating for each new
sampler or scaler. Instead this restores the pre-2.6 default for checkpoints
the project itself produced during the same run.

That is a genuine trust decision, not a workaround to apply blindly: unpickling
executes arbitrary code, so it is only safe because these files are written by
this codebase on this machine. Do not enable it before loading a checkpoint
from an untrusted source.
"""

import functools
from typing import Any

import torch

_PATCHED = False


def allow_full_checkpoint_loading() -> None:
    """Restores ``weights_only=False`` as the default for ``torch.load``.

    Idempotent, so entry points can call it unconditionally. A caller that
    passes ``weights_only=True`` explicitly is still honoured; only the
    undecided cases are changed.

    Note that "undecided" includes an explicit ``weights_only=None``, which is
    what Lightning's checkpoint loader forwards. Treating only the *absent*
    keyword as undecided is not enough — ``lightning.fabric.utilities.cloud_io``
    always passes the argument, so a ``setdefault`` here never fires.
    """
    global _PATCHED
    if _PATCHED:
        return

    original_load = torch.load

    @functools.wraps(original_load)
    def _load(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _load
    _PATCHED = True
