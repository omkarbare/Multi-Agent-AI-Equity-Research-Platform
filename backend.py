"""
Local FastAPI service for the existing three-node equity research graph.

Run: uvicorn backend:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from equity_research_groq import build_graph

app = FastAPI(title="Equity Research API", version="1.0.0")
graph = build_graph()


class ResearchRequest(BaseModel):
    """Validate the ticker before requesting market data or invoking the model."""
    ticker: str = Field(..., examples=["AAPL", "RELIANCE.NS"])

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", ticker):
            raise ValueError("Use a valid Yahoo Finance ticker, e.g. AAPL or RELIANCE.NS")
        return ticker


class ResearchResponse(BaseModel):
    """Only expose report inputs, not secrets or internal graph state."""
    ticker: str
    report: str
    metrics: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    """Cheap liveness check; does not contact Yahoo Finance or Groq."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run the synchronous LangGraph workflow off FastAPI's event loop."""
    try:
        result = await run_in_threadpool(graph.invoke, {"ticker": request.ticker})
    except (ValueError, RuntimeError) as exc:
        # Keep details in server logs; avoid exposing upstream errors or credentials.
        import logging
        logging.getLogger(__name__).exception("Research failed for %s", request.ticker)
        raise HTTPException(status_code=502, detail="Research failed. Check backend logs and try again.") from exc
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Unexpected error for %s", request.ticker)
        raise HTTPException(status_code=500, detail="Unexpected backend error. Check server logs.") from exc
    return ResearchResponse(ticker=request.ticker, report=result["report"], metrics=result["metrics"])
