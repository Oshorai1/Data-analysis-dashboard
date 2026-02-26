from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class KPIResult:
    mode: str
    computed: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, Any] = field(default_factory=dict)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    used_columns: Dict[str, str] = field(default_factory=dict)


@dataclass
class KPIConfig:
    max_items: int = 20
    currency: str = ""  # optional symbol


class KPIEngineBase:
    mode_name: str = "base"
    required_semantic_keys = []

    def __init__(self, config: Optional[KPIConfig] = None):
        self.cfg = config or KPIConfig()

    def compute(self, pipeline_result, config: Optional[KPIConfig] = None) -> KPIResult:
        raise NotImplementedError()
