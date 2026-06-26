"""Dataset adapters for external training frameworks."""

from .registry import (
    AdapterCandidate,
    DataField,
    TargetDataContract,
    choose_adapter_candidate,
    infer_target_data_contract,
)

__all__ = [
    "AdapterCandidate",
    "DataField",
    "TargetDataContract",
    "choose_adapter_candidate",
    "infer_target_data_contract",
]
