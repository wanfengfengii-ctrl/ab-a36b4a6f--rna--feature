"""FastAPI application exposing the versioned folding adjudication endpoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.ensemble import count_ensemble
from app.schemas import (
    EnsembleRequest,
    EnsembleResponse,
    FoldRequest,
    FoldResponse,
    PairCount,
    PositionDistribution,
    Score,
    StructureView,
)
from app.solver import adjudicate, pairs_from_structure

app = FastAPI(
    title="RNA Folding Adjudication Service",
    version="1.0.0",
    description=(
        "Deterministic adjudication of candidate RNA secondary structures via "
        "interval dynamic programming (maximize pairs, then stacked pairs), "
        "plus exact ensemble counting over all legal structures."
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


@app.post(
    "/api/v1/fold/ensemble", response_model=EnsembleResponse, tags=["ensemble"]
)
def fold_ensemble(request: EnsembleRequest) -> EnsembleResponse:
    """Count every legal structure and report per-position distributions."""
    counts = count_ensemble(
        request.sequence,
        forced_positions=request.forced_positions,
        forbidden_positions=request.forbidden_positions,
        positions=request.positions,
    )
    return EnsembleResponse(
        sequence=request.sequence,
        length=len(request.sequence),
        total=str(counts.total),
        positions=[
            PositionDistribution(
                position=dist.position,
                unpaired=str(dist.unpaired),
                pairs=[
                    PairCount(position=partner, count=str(count))
                    for partner, count in dist.pairs
                ],
            )
            for dist in counts.distributions
        ],
    )
