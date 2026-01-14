import time
from typing import Any, Callable, Tuple

import torch


def time_projection(func: Callable, **kwargs) -> Tuple[Any, float]:
    """Run a projection and return its result alongside the wall-clock time.

    CUDA is synchronised on both sides so the measurement covers the kernels
    rather than just the launches.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = func(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return result, time.perf_counter() - t0
