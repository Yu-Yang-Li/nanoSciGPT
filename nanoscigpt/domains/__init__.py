"""Built-in domain metadata and adapters."""

from .registry import (
    DOMAIN_REGISTRY,
    DOMAIN_SPECS,
    RUNNABLE_DOMAINS,
    SEQUENCE_DOMAINS,
    STRUCTURED_DOMAINS,
    DomainSpec,
    get_domain_spec,
    is_structured_domain,
)

__all__ = [
    "DOMAIN_REGISTRY",
    "DOMAIN_SPECS",
    "RUNNABLE_DOMAINS",
    "SEQUENCE_DOMAINS",
    "STRUCTURED_DOMAINS",
    "DomainSpec",
    "get_domain_spec",
    "is_structured_domain",
]
