"""Role-specific policies, shared planning code, and the agent registry.

The package root exposes only the two primary baselines and the registries.
Import advanced policies from their owning role module.
"""

from .fugitive.bir import BeliefInformedRandomFugitiveAgent
from .fugitive.hierarchical_random import HierarchicalRandomFugitiveAgent
from .marshal.bir.bootstrap import BeliefInformedRandomMarshalAgent
from .marshal.hierarchical_random import HierarchicalRandomMarshalAgent
from .registry import (
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    FUGITIVE_AGENT_REGISTRY,
    MARSHAL_AGENT_REGISTRY,
)

__all__ = [
    "BeliefInformedRandomFugitiveAgent",
    "BeliefInformedRandomMarshalAgent",
    "DEFAULT_FUGITIVE_AGENT",
    "DEFAULT_MARSHAL_AGENT",
    "FUGITIVE_AGENT_REGISTRY",
    "HierarchicalRandomFugitiveAgent",
    "HierarchicalRandomMarshalAgent",
    "MARSHAL_AGENT_REGISTRY",
]
