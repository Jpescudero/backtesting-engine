from __future__ import annotations

from importlib.metadata import EntryPoints, entry_points
from typing import Callable, Dict, Generic, Iterable, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Registro simple de fábricas (estrategias, feeds, brokers, etc.).

    Permite registrar objetos por nombre y recuperarlos dinámicamente,
    facilitando configuraciones cargadas desde archivos o CLI.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator/función para registrar una fábrica.

        Si ``name`` es ``None`` se usa el nombre de la clase/función.
        Los nombres se almacenan en minúsculas para evitar problemas
        de mayúsculas/minúsculas.
        """

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            key = (name or factory.__name__).lower()
            self._items[key] = factory
            return factory

        return decorator

    def add(self, name: str, factory: Callable[..., T]) -> None:
        """Registrar una fábrica usando API imperativa."""

        self._items[name.lower()] = factory

    def get(self, name: str) -> Callable[..., T]:
        key = name.lower()
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<vacío>"
            raise KeyError(
                f"No se encontró {self.kind} con nombre '{name}'. Disponible: {available}"
            )
        return self._items[key]

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> Iterable[str]:
        return sorted(self._items)

    def load_entrypoints(self, group: str) -> None:
        """Carga plugins registrados vía entry points.

        - ``group``: nombre del grupo en setup.cfg/pyproject (e.g.,
          ``backtesting_engine.strategies``).
        """

        eps: EntryPoints = entry_points()
        selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])

        for ep in selected:
            if ep.name in self._items:
                continue
            self._items[ep.name.lower()] = ep.load()


strategy_registry: Registry[object] = Registry("estrategia")
feed_registry: Registry[object] = Registry("feed")
broker_registry: Registry[object] = Registry("broker")
cost_model_registry: Registry[object] = Registry("modelo de costos")


ENTRYPOINT_GROUPS = {
    "strategies": "backtesting_engine.strategies",
    "feeds": "backtesting_engine.feeds",
    "brokers": "backtesting_engine.brokers",
    "cost_models": "backtesting_engine.cost_models",
}


def load_plugin_entrypoints() -> None:
    """Carga todos los plugins declarados via entry points."""

    strategy_registry.load_entrypoints(ENTRYPOINT_GROUPS["strategies"])
    feed_registry.load_entrypoints(ENTRYPOINT_GROUPS["feeds"])
    broker_registry.load_entrypoints(ENTRYPOINT_GROUPS["brokers"])
    cost_model_registry.load_entrypoints(ENTRYPOINT_GROUPS["cost_models"])
