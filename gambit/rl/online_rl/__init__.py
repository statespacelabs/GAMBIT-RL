"""Phase 3A: Online RL modules.

Provides telemetry-based PPO actor-critic initialized via distillation
from the Phase 2 IQL teacher.
"""

from .telemetry_encoder import TelemetryEncoder
from .action_schema import (
    ActionCommandConverter,
    CORE_ACTION_INDICES,
    CORE_ACTION_NAMES,
)
from .core_action_projector import CoreActionProjector
from .hybrid_distribution import HybridDistribution
from .actor_critic import RecurrentActorCritic
from .visual_actor_critic import VisualRecurrentActorCritic, VisualNudgeFusion

__all__ = [
    "ActionCommandConverter",
    "CORE_ACTION_INDICES",
    "CORE_ACTION_NAMES",
    "CoreActionProjector",
    "HybridDistribution",
    "RecurrentActorCritic",
    "TelemetryEncoder",
    "VisualRecurrentActorCritic",
    "VisualNudgeFusion",
]
