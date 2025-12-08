"""Herramientas para gestionar y despachar eventos temporales."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Literal
import heapq
import itertools

EventType = Literal["tick", "order_fill", "timer"]


@dataclass(frozen=True)
class Event:
    """Representa un evento generado por el motor.

    Attributes:
        type: Tipo de evento (tick, order_fill, timer).
        timestamp: Momento temporal en el que debe ejecutarse el evento.
        payload: Datos adicionales asociados al evento.
    """

    type: EventType
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


class EventQueue:
    """Cola prioritaria para despachar eventos en orden temporal."""

    def __init__(self, events: Iterable[Event] | None = None) -> None:
        self._heap: list[tuple[float, int, Event]] = []
        self._counter = itertools.count()
        if events is not None:
            for event in events:
                self.push(event)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._heap)

    def push(self, event: Event) -> None:
        """Inserta un evento preservando la estabilidad temporal."""

        heapq.heappush(self._heap, (event.timestamp, next(self._counter), event))

    def pop(self) -> Event:
        """Extrae el siguiente evento en orden temporal."""

        if not self._heap:
            raise IndexError("No hay eventos en la cola")
        _, _, event = heapq.heappop(self._heap)
        return event

    def peek(self) -> Event:
        """Consulta el próximo evento sin retirarlo."""

        if not self._heap:
            raise IndexError("No hay eventos en la cola")
        return self._heap[0][2]

    def dispatch(self, handler: Callable[[Event], None]) -> None:
        """Despacha todos los eventos usando el manejador indicado."""

        while self._heap:
            handler(self.pop())

    def __iter__(self) -> Iterator[Event]:  # pragma: no cover - trivial
        while self._heap:
            yield self.pop()
