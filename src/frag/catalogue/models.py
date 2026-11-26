"""Catalogue value objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogueItem:
    """One product: metadata plus its PNG image (base64), and a score from a search."""

    id: int
    category: str
    description: str
    price: float
    image_b64: str
    score: float | None = None
