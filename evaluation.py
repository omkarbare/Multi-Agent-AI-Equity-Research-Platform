"""Evaluate the actual fetch -> calculate -> write LangGraph workflow with DeepEval.

Run:
    python evaluation.py --tickers AAPL MSFT
    python evaluation.py --tickers RELIANCE.NS
    python evaluation.py --tickers AAPL --threshold 0.7

This invokes Yahoo Finance and Groq for EACH ticker, then Groq again as the
DeepEval judge. It is an end-to-end quality evaluation, not an audited financial
verification or a traced agent-trajectory evaluation. No FastAPI server needed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from equity_research_groq import build_graph

# Work even when this script is launched from a different working directory.
load_dotenv(Path(__file__).resolve().with_name(".env"))


class GroqJudge(DeepEvalBaseLLM):
    """Use the existing Groq account for DeepEval's LLM-as-a-judge.

    Without an explicit judge, DeepEval can default to an OpenAI model; this
    adapter avoids accidentally requiring or charging a second provider.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, temperature=0, max_retries=2, timeout=90)

    def load_model(self) -> ChatGroq:
        """Return the initialized LangChain chat model."""
        return self.model

    def generate(self, prompt: str) -> str:
        """Return a plain string for synchronous DeepEval metrics."""
        return str(self.model.invoke(prompt).content)

    async def a_generate(self, prompt: str) -> str:
        """Support DeepEval's async judge path too."""
        return str((await self.model.ainvoke(prompt)).content)

    def get_model_name(self) -> str:
        """Label the judging model in evaluation output."""
        return f"Groq/{self.model_name}"


def build_metrics(judge: GroqJudge, threshold: float) -> list[GEval]:
    """Define two narrow, evidence-aware rubrics for a research report.

    Explicit evaluation steps avoid extra judge calls to generate steps and
    make the rubric more consistent across runs. The judge is still subjective.
    """
    grounding = GEval(
        name="Evidence grounding",
        evaluation_steps=[
            "Compare every specific company, financial, numerical, period and source claim in actual_output with the evidence in input.",
            "Penalize invented or contradicted figures, fabricated company events, unsupported forecasts and false claims that the data was audited.",
            "Treat unsupported bull/bear narratives as failures unless they are clearly conditional hypotheses grounded in the evidence.",
            "Do not penalize the answer for explicitly saying that a field is missing or a reporting period is unspecified.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=threshold,
        async_mode=False,
    )
    usefulness = GEval(
        name="Research note usefulness",
        evaluation_steps=[
            "Assess whether actual_output provides an understandable company snapshot, financial performance, valuation discussion, bull/bear hypotheses and data gaps for the ticker in input.",
            "Check that price date, source, retrieval time, and known currency are visible; penalize if relevant available values are ignored.",
            "Reward appropriate caveats about approximately one-year price return and unspecified or mismatched provider reporting periods.",
            "Penalize unsupported recommendations, targets and excessive vague filler; do not demand figures that the input marks unavailable.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=threshold,
        async_mode=False,
    )
    return [grounding, usefulness]


def make_test_case(ticker: str, result: dict[str, Any]) -> LLMTestCase:
    """Use the graph's *actual* metrics as the ground truth available to the writer.

    The input explicitly contains JSON evidence, so the grounding judge can
    compare report claims to the information the writer really received.
    This does not independently verify Yahoo Finance against company filings.
    """
    evidence = json.dumps(result["metrics"], indent=2, allow_nan=False, default=str)
    return LLMTestCase(
        input=f"Write an equity research note for {ticker} using only this evidence:\n{evidence}",
        actual_output=result["report"],
    )


def save_results(records: list[dict[str, Any]], output: Path) -> None:
    """Write per-ticker scores and reasons to a local JSON file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def main() -> int:
    """Invoke the full graph, run two DeepEval judges and enforce a threshold."""
    parser = argparse.ArgumentParser(description="DeepEval checks for the equity research graph")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"],
                        help="Symbols to evaluate (default: AAPL MSFT)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Passing score for each metric, between 0 and 1 (default: 0.7)")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Local JSON results path (default: evaluation_results.json)")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    tickers = [ticker.strip().upper() for ticker in args.tickers]
    if any(not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", ticker) for ticker in tickers):
        parser.error("Each ticker must be a valid Yahoo Finance symbol")
    if os.getenv("GROQ_API_KEY") in (None, "", "replace_with_your_groq_api_key"):
        parser.error("Add a real GROQ_API_KEY to the .env beside evaluation.py")

    judge_name = os.getenv("GROQ_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    judge = GroqJudge(judge_name)
    graph = build_graph()
    records: list[dict[str, Any]] = []
    all_passed = True

    # New metrics for each ticker prevent stale state/reasons from leaking
    # between test cases. Sequential runs reduce Groq rate-limit pressure.
    for ticker in tickers:
        print(f"\nEvaluating {ticker} ...", flush=True)
        record: dict[str, Any] = {"ticker": ticker, "judge_model": judge.get_model_name(),
                                  "threshold": args.threshold, "metrics": {}}
        try:
            result = graph.invoke({"ticker": ticker})
            test_case = make_test_case(ticker, result)
            record["source"] = result["metrics"].get("source")
            record["retrieved_utc"] = result["metrics"].get("retrieved_utc")
            for metric in build_metrics(judge, args.threshold):
                metric.measure(test_case)
                score = float(metric.score) if metric.score is not None else None
                passed = score is not None and score >= args.threshold
                record["metrics"][metric.name] = {
                    "score": score, "passed": passed, "reason": metric.reason,
                }
                print(f"  {metric.name}: {score if score is not None else 'N/A'} "
                      f"({'PASS' if passed else 'FAIL'})")
                if metric.reason:
                    print(f"  Reason: {metric.reason}")
                all_passed &= passed
        except Exception as exc:
            # A failed graph/judge run is never silently counted as passing.
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {record['error']}", file=sys.stderr)
            all_passed = False
        records.append(record)
        save_results(records, Path(args.output))

    print(f"\nResults saved to {args.output}")
    print("Overall:", "PASS" if all_passed else "FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
