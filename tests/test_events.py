import pytest

from src.engine.events import Event, EventQueue


def test_events_are_dispatched_in_timestamp_order():
    queue = EventQueue(
        [
            Event(type="tick", timestamp=3.0),
            Event(type="order_fill", timestamp=1.0),
            Event(type="timer", timestamp=2.0),
        ]
    )

    dispatched = []
    queue.dispatch(dispatched.append)

    assert [event.type for event in dispatched] == ["order_fill", "timer", "tick"]
    assert [event.timestamp for event in dispatched] == [1.0, 2.0, 3.0]


def test_late_insertions_are_ordered_correctly():
    queue = EventQueue()
    queue.push(Event(type="tick", timestamp=5.0))
    queue.push(Event(type="order_fill", timestamp=10.0))

    # Ingreso tardío antes de procesar
    queue.push(Event(type="timer", timestamp=7.0))

    processed = [queue.pop(), queue.pop(), queue.pop()]
    assert [event.timestamp for event in processed] == [5.0, 7.0, 10.0]


def test_dispatch_updates_state_consistently():
    queue = EventQueue()
    state = {"processed": 0, "last": None}

    for ts in [2.0, 1.5, 3.5]:
        queue.push(Event(type="tick", timestamp=ts, payload={"ts": ts}))

    def handler(event: Event) -> None:
        state["processed"] += 1
        state["last"] = event.timestamp

    queue.dispatch(handler)

    assert len(queue) == 0
    assert state == {"processed": 3, "last": 3.5}

    with pytest.raises(IndexError):
        queue.pop()
