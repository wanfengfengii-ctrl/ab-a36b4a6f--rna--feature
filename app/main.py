"""FastAPI application exposing the versioned folding adjudication endpoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.schemas import FoldRequest, FoldResponse, Score, StructureView
from app.solver import adjudicate, pairs_from_structure

app = FastAPI(
    title="RNA Folding Adjudication Service",
    version="1.0.0",
    description=(
        "Deterministic adjudication of candidate RNA secondary structures via "
        "interval dynamic programming (maximize pairs, then stacked pairs)."
    ),
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness/readiness probe used by Docker health checks."""
    return {"status": "ok"}


@app.post("/api/v1/fold", response_model=FoldResponse, tags=["adjudication"])
def fold(request: FoldRequest) -> FoldResponse:
    """Adjudicate one constrained RNA folding instance."""
    verdict = adjudicate(
        request.sequence,
        forced_positions=request.forced_positions,
        forbidden_positions=request.forbidden_positions,
    )
    if not verdict.feasible:
        return FoldResponse(
            status="INFEASIBLE",
            sequence=request.sequence,
            length=len(request.sequence),
            unique=False,
            primary=None,
            witness=None,
        )

    def view(structure: str) -> StructureView:
        return StructureView(
            structure=structure,
            pairs=pairs_from_structure(structure),
            score=Score(pairs=verdict.pairs, stacks=verdict.stacks),
        )

    return FoldResponse(
        status="OPTIMAL",
        sequence=request.sequence,
        length=len(request.sequence),
        unique=verdict.witness is None,
        primary=view(verdict.primary),
        witness=view(verdict.witness) if verdict.witness is not None else None,
    )
