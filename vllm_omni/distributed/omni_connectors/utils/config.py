# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

from .logging import get_connector_logger

logger = get_connector_logger(__name__)

TRANSFER_ENGINE_CONNECTOR_NAMES = frozenset(
    {
        "MooncakeTransferEngineConnector",
        "MoriTransferEngineConnector",
        "YuanrongTransferEngineConnector",
    }
)


@dataclass
class ConnectorSpec:
    """Specification for a connector instance."""

    name: str  # e.g., "MooncakeStoreConnector", "SharedMemoryConnector", "YuanrongConnector"
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific config


def stage_connector_extra(connector_cfg: Any) -> dict[str, Any]:
    """Extract the connector ``extra`` from a stage_connector_config of either shape:
    legacy ``{"name","extra"}`` or dual ``{"input":{...},"output":{...}}`` (extras merged).
    Tolerates a non-dict (object with ``.extra``) and missing keys; returns ``{}`` if absent."""
    if connector_cfg is None:
        return {}
    if not isinstance(connector_cfg, dict):
        extra = getattr(connector_cfg, "extra", None)
        return extra if isinstance(extra, dict) else {}
    if "input" in connector_cfg or "output" in connector_cfg:
        merged: dict[str, Any] = {}
        for direction in ("input", "output"):
            sub = connector_cfg.get(direction)
            if isinstance(sub, dict) and isinstance(sub.get("extra"), dict):
                merged.update(sub["extra"])
        return merged
    extra = connector_cfg.get("extra")
    return extra if isinstance(extra, dict) else {}


@dataclass
class OmniTransferConfig:
    """
    Top-level configuration for OmniConnector system.
    Members:
        connectors: A dictionary of connectors, keyed by (from_stage, to_stage).
        default_connector: The default connector to use if no connector is specified for an edge.
    """

    # Direct mapping: (from_stage, to_stage) -> connector
    connectors: dict[tuple[str, str], ConnectorSpec] = field(default_factory=dict)
    default_connector: ConnectorSpec | None = None

    def get_connector_for_edge(self, from_stage: str, to_stage: str) -> ConnectorSpec | None:
        """Get connector spec for a specific edge."""
        edge_key = (from_stage, to_stage)
        return self.connectors.get(edge_key, self.default_connector)

    def has_connector_for_edge(self, from_stage: str, to_stage: str) -> bool:
        """Check if there's a connector configured for the edge."""
        return self.get_connector_for_edge(from_stage, to_stage) is not None
