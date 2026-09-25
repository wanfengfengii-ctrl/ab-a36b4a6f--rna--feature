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
