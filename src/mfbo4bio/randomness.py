from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SeedManager:
    base_seed: int

    def _derive_seed(self, namespace: str) -> int:
        payload = f"{self.base_seed}:{namespace}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:16], 16) % (2**32 - 1)

    def child(self, namespace: str) -> "SeedManager":
        return SeedManager(self._derive_seed(namespace))

    def numpy_rng(self, namespace: str = "numpy") -> np.random.Generator:
        return np.random.default_rng(self._derive_seed(namespace))

    def torch_seed(self, namespace: str = "torch") -> int:
        seed = self._derive_seed(namespace)
        torch.manual_seed(seed)
        return seed

    def sobol_seed(self, namespace: str = "sobol") -> int:
        return self._derive_seed(namespace)
