"""Queue-based matchmaking: pair by closest rating, with a window that widens
the longer a player has waited.

Algorithm (per tick):
1. Shuffle the queue (so ties, e.g. everyone at 1500 on day one, break randomly),
   then stable-sort by rating.
2. Walk the sorted list. Two neighbours are matched if their rating gap is
   within the window of whichever of them has waited longer; otherwise the
   lower one is skipped and stays in the queue.

Greedy neighbour pairing is what most real systems approximate, and it is
O(n log n). It is not a globally optimal assignment, which is fine here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WideningWindow:
    base: float = 50.0
    growth: float = 25.0
    max: float = 600.0

    def __call__(self, wait_ticks) -> np.ndarray:
        return np.minimum(self.base + self.growth * np.asarray(wait_ticks, dtype=float), self.max)


def pair_random(queued: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pair queued players uniformly at random (odd one out keeps waiting)."""
    queued = np.asarray(queued, dtype=np.int64)
    ids = queued[rng.permutation(queued.size)]
    m = ids.size // 2
    return ids[:m].copy(), ids[m:2 * m].copy()


def pair_players(
    queued: np.ndarray,
    ratings: np.ndarray,
    waits: np.ndarray,
    window: WideningWindow,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two equal-length arrays of player ids (A side, B side).

    ``queued`` holds player ids in the queue; ``ratings`` and ``waits`` are
    indexed by player id over the whole population.
    """
    queued = np.asarray(queued, dtype=np.int64)
    if queued.size < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    shuffled = queued[rng.permutation(queued.size)]
    order = np.argsort(ratings[shuffled], kind="stable")
    ids = shuffled[order]

    r = ratings[ids]
    gaps = np.diff(r).tolist()
    allow = window(waits[ids])
    allow_pair = np.maximum(allow[:-1], allow[1:]).tolist()

    a_out: list[int] = []
    b_out: list[int] = []
    ids_list = ids.tolist()
    i, n = 0, len(ids_list)
    while i < n - 1:
        if gaps[i] <= allow_pair[i]:
            a_out.append(ids_list[i])
            b_out.append(ids_list[i + 1])
            i += 2
        else:
            i += 1

    a = np.asarray(a_out, dtype=np.int64)
    b = np.asarray(b_out, dtype=np.int64)
    # Randomise which side is "A" so side never correlates with rating.
    swap = rng.random(a.size) < 0.5
    a[swap], b[swap] = b[swap], a[swap].copy()
    return a, b
