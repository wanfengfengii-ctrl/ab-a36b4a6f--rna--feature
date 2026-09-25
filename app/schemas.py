"""Request/response schemas for the versioned folding adjudication API.

All validation lives here so that malformed input is rejected with HTTP 422
before the solver is ever invoked.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

MIN_LENGTH = 20
MAX_LENGTH = 240
VALID_BASES = frozenset("ACGU")

# The ensemble endpoint supports a narrower length range than adjudication.
ENSEMBLE_MIN_LENGTH = 20
ENSEMBLE_MAX_LENGTH = 120
#: A request may ask for the distribution of at most this many positions.
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

    * ``sequence``: RNA sequence, 20-120 bases for this mode, alphabet ACGU
      (case-insensitive, normalized to upper case).
    * ``forced_positions`` / ``forbidden_positions``: same 0-based constraint
      inputs as adjudication (deduplicated and sorted).
    * ``positions``: 1-12 distinct 0-based review positions whose pairing
      distributions are reported (sorted ascending for a canonical response).
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str
    forced_positions: list[StrictInt] = []
    forbidden_positions: list[StrictInt] = []
    positions: list[StrictInt]

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

    @field_validator("positions")
    @classmethod
    def _validate_positions(cls, value: list[int]) -> list[int]:
        if not 1 <= len(value) <= MAX_REVIEW_POSITIONS:
            raise ValueError(
                f"positions must contain between 1 and {MAX_REVIEW_POSITIONS} "
                f"entries, got {len(value)}"
            )
        if len(set(value)) != len(value):
            raise ValueError("positions contains duplicate entries")
        return sorted(value)

    @model_validator(mode="after")
    def _validate_positions_in_range(self) -> "EnsembleRequest":
        n = len(self.sequence)
        for name in ("forced_positions", "forbidden_positions", "positions"):
            for pos in getattr(self, name):
                if pos < 0 or pos >= n:
                    raise ValueError(
                        f"{name} contains out-of-range position {pos} "
                        f"for sequence length {n}"
                    )
        # Deterministic normalization, matching adjudication semantics.
        self.forced_positions = sorted(set(self.forced_positions))
        self.forbidden_positions = sorted(set(self.forbidden_positions))
        return self


class PartnerCount(BaseModel):
    """Structures pairing the review position with one partner position."""

    position: int
    count: str  # exact count as a decimal string


class PositionDistribution(BaseModel):
    """Pairing distribution for one review position.

    ``unpaired`` plus the sum of all ``partners`` counts always equals the
    ensemble total.  When no legal structure exists, ``unpaired`` is "0" and
    ``partners`` is empty.
    """

    position: int
    unpaired: str  # exact count as a decimal string
    partners: list[PartnerCount]


class EnsembleResponse(BaseModel):
    """Exact counts over all legal structures of one constrained instance."""

    status: Literal["FEASIBLE", "INFEASIBLE"]
    sequence: str
    length: int
    total: str  # total number of legal structures, as a decimal string
    positions: list[PositionDistribution]
