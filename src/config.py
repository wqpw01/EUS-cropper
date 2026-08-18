"""Shared coordinate model for the fixed 10 cm EUS crop."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CropRegion:
    """Pixel-aligned rectangular crop region using PIL half-open bounds."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)
