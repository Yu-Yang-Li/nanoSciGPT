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
    model_unit: str
    preserved_relations: str
    pretraining_objective: str
    task_name: str
    downstream_training: str
    source_kind: str


def build_registry(specs: Iterable[DomainSpec]):
    registry = {}
    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"duplicate domain: {spec.name}")
        registry[spec.name] = spec
    return MappingProxyType(registry)


DOMAIN_SPECS = (
    DomainSpec(
        "text", "sequence", "character_tokens", "字符", "字符顺序和上下文",
        "预测下一个字符", "punctuation-density teaching classification", "full_fine_tune", "public_source",
    ),
    DomainSpec(
        "protein", "sequence", "amino_acid_tokens", "氨基酸", "序列顺序和每条序列的边界",
        "预测下一个氨基酸", "protein composition teaching classification", "frozen_probe", "public_source",
    ),
    DomainSpec(
        "dna", "sequence", "nucleotide_tokens", "碱基", "局部基序和长区段顺序",
        "预测下一个碱基", "DNA GC-content teaching classification", "frozen_probe", "public_source",
    ),
    DomainSpec(
        "smiles", "sequence", "SMILES_tokens", "SMILES标记", "原子和键在线性记法中的顺序",
        "预测下一个SMILES标记", "ESOL aqueous-solubility teaching regression", "frozen_probe", "public_source",
    ),
    DomainSpec(
        "weather", "structured", "spatiotemporal_patches", "时空网格块", "空间邻域和时间先后",
        "重建被遮住的时空网格块", "advection speed teaching regression", "frozen_probe", "synthetic_fixture",
    ),
    DomainSpec(
        "crystal", "structured", "periodic_graph", "原子与周期邻域", "晶格周期、距离和邻接",
        "判断被遮住位置的原子种类", "unit-cell mass density proxy regression", "frozen_probe", "synthetic_fixture",
    ),
    DomainSpec(
        "structure3d", "structured", "pairwise_distance_tokens", "点之间的距离", "三维邻近关系和刚体变换不变性",
        "重建被遮住的距离片段", "helix pitch teaching regression", "frozen_probe", "synthetic_fixture",
    ),
    DomainSpec(
        "image", "structured", "image_patches", "二维图像块", "二维空间邻域",
        "重建被遮住的图像块", "astronomical source-count teaching regression", "frozen_probe", "synthetic_fixture",
    ),
    DomainSpec(
        "spectrum", "structured", "wavelength_patches", "连续波长区间", "波长顺序和局部谱线形状",
        "重建被遮住的连续波长区间", "blackbody temperature teaching regression", "frozen_probe", "synthetic_fixture",
    ),
    DomainSpec(
        "field", "structured", "space_time_patches", "时空场片段", "空间邻域、时间变化和边界",
        "重建被遮住的场片段", "diffusion coefficient teaching regression", "frozen_probe", "synthetic_fixture",
    ),
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
