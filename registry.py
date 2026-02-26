from typing import Dict
from .base import KPIEngineBase, KPIResult, KPIConfig

_REGISTRY: Dict[str, KPIEngineBase] = {}

def register(name: str, engine_cls):
    _REGISTRY[name] = engine_cls
    return engine_cls

def get_engine(name: str) -> KPIEngineBase:
    name = (name or 'universal').lower()
    cls = _REGISTRY.get(name)
    if cls:
        return cls()
    # fallback to universal
    cls = _REGISTRY.get('universal')
    if cls:
        return cls()
    raise KeyError('No KPI engine registered')


# register default engines lazily to avoid import cycles
try:
    from .universal import UniversalKPIEngine
    register('universal', UniversalKPIEngine)
except Exception:
    pass
try:
    from .clinic import ClinicKPIEngine
    register('clinic', ClinicKPIEngine)
except Exception:
    pass
try:
    from .trading import TradingKPIEngine
    register('trading', TradingKPIEngine)
except Exception:
    pass
