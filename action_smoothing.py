"""Smooth a policy's action chunks before executing them (step4_eval.py --smooth).

Two lag-free fixes for the jitter measured by diag_smoothness.py:
  * smooth_chunk: Gaussian filter along time inside one predicted chunk. The chunk is a plan for the
    next 50 steps, so filtering it (centred window) does not delay anything.
  * crossfade: the new chunk predicted after executing n steps lines up with the old chunk's steps n...;
    blend from the old plan into the new one over `steps` steps instead of jumping.
Arm joints only: grippers (dims 6 and 13) are left as predicted so grasp timing does not change.
numpy only, so it works in both venvs.
"""
import numpy as np

ARM = np.array(list(range(6)) + list(range(7, 13)))
SIGMA = 2.0       # steps
CROSSFADE = 10    # steps


def smooth_chunk(chunk, sigma=SIGMA):
    """chunk: (T, 14). Returns a copy with arm joints Gaussian-filtered along time (edges held)."""
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    out = chunk.copy()
    padded = np.pad(chunk[:, ARM], ((radius, radius), (0, 0)), mode="edge")
    out[:, ARM] = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(len(ARM))], axis=1)
    return out


def crossfade(new_chunk, old_tail, steps=CROSSFADE):
    """Blend old_tail (old chunk's not-yet-executed steps, aligned with new_chunk[0:]) into new_chunk."""
    n = min(steps, len(old_tail), len(new_chunk))
    if n == 0:
        return new_chunk
    out = new_chunk.copy()
    w = (np.arange(1, n + 1) / (n + 1))[:, None]  # weight of the new plan: ~0 -> ~1
    out[:n, ARM] = (1 - w) * old_tail[:n, ARM] + w * new_chunk[:n, ARM]
    return out


class ChunkSmoother:
    """Per-episode state: call reset() at episode start, then next(chunk, n_exec) for each new chunk."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev, self._n_exec = None, 0

    def next(self, chunk, n_exec):
        chunk = smooth_chunk(np.asarray(chunk))
        if self._prev is not None:
            chunk = crossfade(chunk, self._prev[self._n_exec:])
        self._prev, self._n_exec = chunk, n_exec
        return chunk
