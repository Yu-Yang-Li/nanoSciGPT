"""Single source of truth for the bundled classroom domains."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable


@dataclass(frozen=True)
class DomainSpec:
    name: str
    family: str
    representation: str
    task_name: str
    source_kind: str


def build_registry(specs: Iterable[DomainSpec]):
    registry = {}
    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"duplicate domain: {spec.name}")
        registry[spec.name] = spec
    return MappingProxyType(registry)


DOMAIN_SPECS = (
    DomainSpec("text", "sequence", "character_tokens", "punctuation-density teaching classification", "public_source"),
    DomainSpec("protein", "sequence", "amino_acid_tokens", "protein composition teaching classification", "public_source"),
    DomainSpec("dna", "sequence", "nucleotide_tokens", "DNA GC-content teaching classification", "public_source"),
    DomainSpec("smiles", "sequence", "SMILES_tokens", "ESOL aqueous-solubility teaching regression", "public_source"),
    DomainSpec("weather", "structured", "spatiotemporal_patches", "advection speed teaching regression", "synthetic_fixture"),
    DomainSpec("crystal", "structured", "periodic_graph", "unit-cell mass density proxy regression", "synthetic_fixture"),
    DomainSpec("structure3d", "structured", "pairwise_distance_tokens", "helix pitch teaching regression", "synthetic_fixture"),
    DomainSpec("image", "structured", "image_patches", "astronomical source-count teaching regression", "synthetic_fixture"),
    DomainSpec("spectrum", "structured", "wavelength_patches", "blackbody temperature teaching regression", "synthetic_fixture"),
    DomainSpec("field", "structured", "space_time_patches", "diffusion coefficient teaching regression", "synthetic_fixture"),
)

DOMAIN_REGISTRY = build_registry(DOMAIN_SPECS)
SEQUENCE_DOMAINS = tuple(spec.name for spec in DOMAIN_SPECS if spec.family == "sequence")
STRUCTURED_DOMAINS = tuple(spec.name for spec in DOMAIN_SPECS if spec.family == "structured")
RUNNABLE_DOMAINS = SEQUENCE_DOMAINS + STRUCTURED_DOMAINS


def get_domain_spec(name: str) -> DomainSpec:
    try:
        return DOMAIN_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown classroom domain: {name}") from error


def is_structured_domain(name: str) -> bool:
    return get_domain_spec(name).family == "structured"
