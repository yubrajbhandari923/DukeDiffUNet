# utils/profiling.py

import time
import functools
import aim
from torch.profiler import profile, record_function, ProfilerActivity
import torch

_GLOBAL_PROF_CFG = None


def init_profiler(cfg, aimRun, rank=0):
    global _GLOBAL_PROF_CFG, _GLOBAL_PROF_RANK, _GLOBAL_AIM_RUN
    _GLOBAL_PROF_CFG = cfg
    _GLOBAL_PROF_RANK = rank
    _GLOBAL_AIM_RUN = aimRun


def profile_block(name):
    """
    Decorator to time any function and log to Aim under context 'timing'.
    No-op if cfg.profile is False.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            cfg = _GLOBAL_PROF_CFG
            rank = _GLOBAL_PROF_RANK
            
            if rank == 0:  # Only profile on rank 0
                if cfg is None:
                    raise ValueError("Profiler not initialized. Call init_profiler(cfg) first.")

                if not getattr(cfg, "profile", False):
                    return fn(*args, **kwargs)
                start = time.time()
                result = fn(*args, **kwargs)
                elapsed = time.time() - start
                # run = aim.Run.active_run()
                run = _GLOBAL_AIM_RUN
                if run:
                    run.track(float(elapsed), name=name, context={"type": "timing"})
                else:
                    print(f"Warning: No active Aim run found. Profiling data for '{name}' not logged.")
                return result
            else:
                # If not rank 0, just call the function without profiling
                return fn(*args, **kwargs)

        return wrapper
    return decorator


class TorchProfiler:
    """
    Context manager to wrap PyTorch/CUDA profiling.
    Only activates when cfg.profile is True.
    Dumps TensorBoard traces into logdir/{subdir}.
    """

    def __init__(self, subdir="torch_profile"):
        self.cfg = _GLOBAL_PROF_CFG

        if self.cfg is None:
            raise ValueError("Profiler not initialized. Call init_profiler(cfg) first.")

        self.subdir = subdir
        self.prof = None

    def __enter__(self):
        if not getattr(self.cfg, "torch_profile", False):
            return None
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=f"./logdir/{self.subdir}/rank{_GLOBAL_PROF_RANK}"
            ),
            record_shapes=True,
            with_stack=True,
        )
        self.prof.__enter__()
        return self.prof

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prof:
            self.prof.__exit__(exc_type, exc_val, exc_tb)
