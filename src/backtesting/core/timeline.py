from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from src.backtesting.core.models import MarketDataBatch


@dataclass
class Timeline:
    """Lightweight iterator over a :class:`MarketDataBatch`.

    The timeline isolates sequencing logic from strategies and brokers, making
    it easier to swap implementations while preserving the iteration contract.
    """

    data: MarketDataBatch

    def iter_indices(self) -> Iterator[int]:
        for i in range(self.data.size):
            yield i

    def iter_slices(self, window: int) -> Iterator[MarketDataBatch]:
        for start in range(0, self.data.size, window):
            yield self.data.slice_from(start)


def ensure_sequential_orders(order_batches: Iterable[Sequence[Any]]) -> bool:
    """Helper used in tests to ensure strategies emit ordered batches."""

    last_index = -1
    for batch in order_batches:
        for order in batch:
            if order.bar_index < last_index:
                return False
            last_index = order.bar_index
    return True
