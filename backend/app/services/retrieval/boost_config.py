from dataclasses import dataclass


@dataclass(frozen=True)
class BoostConfig:
    """Immutable boost signals set by QAToolExecutor, consumed by reranker.

    All fields default to False/0 — no boost applied unless explicitly enabled.
    """

    front_matter_boost: bool = False       # P1.3
    lead_paragraph_boost: bool = False     # P2.3
    lead_page_threshold: int = 0           # P2.3: absolute page number threshold
