"""Request/response schemas for the versioned folding adjudication API.

All validation lives here so that malformed input is rejected with HTTP 422
before the solver is ever invoked.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

MIN_LENGTH = 20
MAX_LENGTH = 240
VALID_BASES = frozenset("ACGU")

# The ensemble-counting mode deliberately supports a narrower length range
# than plain adjudication.
ENSEMBLE_MIN_LENGTH = 20
ENSEMBLE_MAX_LENGTH = 120
#: How many positions may be reviewed in one ensemble request.
MAX_REVIEW_POSITIONS = 12


class FoldRequest(BaseModel):
    """One adjudication request.

    * ``sequence``: RNA sequence, 20-240 bases, alphabet ACGU (case-insensitive,
      normalized to upper case).
    * ``forced_positions``: 0-based positions that must be paired.
    * ``forbidden_positions``: 0-based positions that must stay unpaired.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str
    forced_positions: list[StrictInt] = []
    forbidden_positions: list[StrictInt] = []

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        seq = value.upper()
        if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            raise ValueError(
                f"sequence length must be between {MIN_LENGTH} and {MAX_LENGTH}, "
                f"got {len(seq)}"
            )
        bad = sorted(set(seq) - VALID_BASES)
        if bad:
            raise ValueError(f"sequence contains invalid bases: {bad}")
        return seq

    @model_validator(mode="after")
    def _validate_positions(self) -> "FoldRequest":
        n = len(self.sequence)
        for name in ("forced_positions", "forbidden_positions"):
            values = getattr(self, name)
            for pos in values:
                if pos < 0 or pos >= n:
                    raise ValueError(
                        f"{name} contains out-of-range position {pos} "
                        f"for sequence length {n}"
                    )
            # Deterministic normalization: deduplicate and sort.
            setattr(self, name, sorted(set(values)))
        return self


class Score(BaseModel):
    """Two-level score: pair count first, then adjacent stacked pairs."""

    pairs: int
    stacks: int


class StructureView(BaseModel):
    """One optimal structure: dot-bracket, pair table, and its score."""

    structure: str
    pairs: list[list[int]]
    score: Score


class FoldResponse(BaseModel):
    """Deterministic adjudication verdict."""

    status: Literal["OPTIMAL", "INFEASIBLE"]
    sequence: str
    length: int
    unique: bool
    primary: StructureView | None
    witness: StructureView | None


class EnsembleRequest(BaseModel):
    """One ensemble-counting request.

    Same sequence/constraint inputs as adjudication, but with this mode's
    own length range (20-120) plus 1-12 distinct positions to review.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str
    forced_positions: list[StrictInt] = []
    forbidden_positions: list[StrictInt] = []
    positions: list[StrictInt] = Field(
        min_length=1, max_length=MAX_REVIEW_POSITIONS
    )

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        seq = value.upper()
        if not ENSEMBLE_MIN_LENGTH <= len(seq) <= ENSEMBLE_MAX_LENGTH:
            raise ValueError(
                f"sequence length must be between {ENSEMBLE_MIN_LENGTH} and "
                f"{ENSEMBLE_MAX_LENGTH}, got {len(seq)}"
            )
        bad = sorted(set(seq) - VALID_BASES)
        if bad:
            raise ValueError(f"sequence contains invalid bases: {bad}")
        return seq

    @model_validator(mode="after")
    def _validate_positions(self) -> "EnsembleRequest":
        n = len(self.sequence)
        for name in ("forced_positions", "forbidden_positions"):
            values = getattr(self, name)
            for pos in values:
                if pos < 0 or pos >= n:
                    raise ValueError(
                        f"{name} contains out-of-range position {pos} "
                        f"for sequence length {n}"
                    )
            # Deterministic normalization: deduplicate and sort.
            setattr(self, name, sorted(set(values)))
        # Reviewed positions: range-checked, and duplicates are rejected
        # outright (never silently deduplicated).
        seen: set[int] = set()
        for pos in self.positions:
            if pos < 0 or pos >= n:
                raise ValueError(
                    f"positions contains out-of-range position {pos} "
                    f"for sequence length {n}"
                )
            if pos in seen:
                raise ValueError(f"positions contains duplicate position {pos}")
            seen.add(pos)
        return self


class PairCount(BaseModel):
    """Structures in which the reviewed position pairs with ``position``."""

    position: int
    count: str


class PositionDistribution(BaseModel):
    """Exact ensemble distribution for one reviewed position.

    ``pairs`` is sorted by partner position ascending; ``unpaired`` plus
    the sum of every pair count equals the ensemble total.
    """

    position: int
    unpaired: str
    pairs: list[PairCount]


class EnsembleResponse(BaseModel):
    """Exact counts over all legal structures, as decimal strings.

    When no legal structure exists, ``total`` is ``"0"`` and every
    reviewed position reports an empty distribution.
    """

    sequence: str
    length: int
    total: str
    positions: list[PositionDistribution]
