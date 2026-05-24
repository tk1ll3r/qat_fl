from __future__ import annotations

import math


class AdaptiveBitScheduler:
    def __init__(self, initial_bits: int, max_bits: int):
        if initial_bits < 1 or max_bits < initial_bits:
            raise ValueError("expected 1 <= initial_bits <= max_bits")
        self.initial_bits = initial_bits
        self.max_bits = max_bits
        self.initial_loss: float | None = None

    def update(self, current_loss: float) -> int:
        current_loss = max(float(current_loss), 1e-12)
        if self.initial_loss is None:
            self.initial_loss = current_loss
            return self.initial_bits
        initial_levels = 2**self.initial_bits
        levels = math.sqrt(self.initial_loss / current_loss) * initial_levels
        bits = math.ceil(math.log2(max(levels, 2.0)))
        return min(max(bits, self.initial_bits), self.max_bits)

