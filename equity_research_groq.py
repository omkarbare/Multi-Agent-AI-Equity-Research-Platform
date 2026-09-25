"""
Minimal equity research pipeline: fetch -> calculate -> write.

"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import yfinance as yf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

# Read the .env next to this file, even when launched from another directory.
load_dotenv(Path(__file__).resolve().with_name(".env"))


class ResearchState(TypedDict, total=False):
    """Values passed from one LangGraph node to the next."""
    ticker: str
    data: dict[str, Any]
    metrics: dict[str, Any]
    report: str


def safe_number(value: Any) -> float | None:
    """Return a JSON-compatible number, or None for missing/invalid values."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_data(state: ResearchState) -> dict[str, Any]:
    """Collect one year of prices and selected company data from Yahoo Finance."""
    ticker = state["ticker"].strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", ticker):
        raise ValueError("Invalid ticker. Try AAPL or RELIANCE.NS.")

    stock = yf.Ticker(ticker)
    try:
        history = stock.history(period="1y", auto_adjust=True, actions=False)
        closes = history["Close"].dropna()
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve prices for {ticker}: {exc}") from exc
    if closes.empty:
        raise ValueError(f"No price history for {ticker}; check the symbol.")

    # Company info is useful but is not always available for every ticker.
    warning = None
    try:
        info = stock.info or {}
    except Exception as exc:
        info = {}
        warning = f"Company information unavailable: {type(exc).__name__}: {exc}"

    fields = (
        "currency", "sector", "industry", "marketCap", "totalRevenue",
        "netIncomeToCommon", "profitMargins", "trailingPE", "forwardPE",
        "priceToBook", "totalDebt", "totalCash", "freeCashflow",
        "revenueGrowth", "earningsGrowth",
    )
    data = {
        "company": info.get("longName") or info.get("shortName") or ticker,
        "source": f"https://finance.yahoo.com/quote/{ticker}/",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_close_date": str(closes.index[0].date()),
        "last_close_date": str(closes.index[-1].date()),
        "first_adjusted_close": safe_number(closes.iloc[0]),
        "last_adjusted_close": safe_number(closes.iloc[-1]),
        "price_observations": len(closes),
        "info": {name: info.get(name) for name in fields},
        "warning": warning,
    }
    return {"ticker": ticker, "data": data}


def calculate(state: ResearchState) -> dict[str, Any]:
    """Compute a few transparent ratios without trusting the LLM with arithmetic."""
    data = state["data"]
    info = data["info"]
    last = data["last_adjusted_close"]
    first = data["first_adjusted_close"]
    revenue = safe_number(info.get("totalRevenue"))
    net_income = safe_number(info.get("netIncomeToCommon"))
    debt = safe_number(info.get("totalDebt"))
    cash = safe_number(info.get("totalCash"))
    profit_margin = safe_number(info.get("profitMargins"))
    revenue_growth = safe_number(info.get("revenueGrowth"))
    earnings_growth = safe_number(info.get("earningsGrowth"))

    # Yahoo's info fields can represent different reporting periods. Do not
    # pretend that a ratio computed from two fields is period-matched.
    metrics = {
        "ticker": state["ticker"],
        "company": data["company"],
        "currency": info.get("currency"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "last_adjusted_close": last,
        "last_close_date": data["last_close_date"],
        "first_close_date": data["first_close_date"],
        "price_observations": data["price_observations"],
        "price_return_available_window_pct": (
            round(100 * (last / first - 1), 2)
            if last is not None and first is not None and first > 0 and data["price_observations"] > 1
            else None
        ),
        "market_cap": safe_number(info.get("marketCap")),
        "revenue_reported_by_provider": revenue,
        "net_income_reported_by_provider": net_income,
        "profit_margin_pct_reported_by_provider": (
            round(100 * profit_margin, 2) if profit_margin is not None else None
        ),
        "trailing_pe": safe_number(info.get("trailingPE")),
        "forward_pe": safe_number(info.get("forwardPE")),
        "price_to_book": safe_number(info.get("priceToBook")),
        "free_cash_flow_reported_by_provider": safe_number(info.get("freeCashflow")),
        "revenue_growth_pct_reported_by_provider": (
            round(100 * revenue_growth, 2) if revenue_growth is not None else None
        ),
        "earnings_growth_pct_reported_by_provider": (
            round(100 * earnings_growth, 2) if earnings_growth is not None else None
        ),
        "net_debt_from_provider_fields": debt - cash if debt is not None and cash is not None else None,
        "period_caveat": "Yahoo company-info metrics may have different or unspecified periods; verify against filings.",
        "warning": data["warning"],
        "retrieved_utc": data["retrieved_utc"],
        "source": data["source"],
    }
    return {"metrics": metrics}


def write_report(state: ResearchState) -> dict[str, str]:
    """Ask Groq for a concise research note grounded only in calculated inputs."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "replace_with_your_groq_api_key":
        raise RuntimeError("Set GROQ_API_KEY in .env (run python env.py first).")

    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        max_retries=2,
        timeout=60,
    )
    evidence = json.dumps(state["metrics"], indent=2, allow_nan=False, default=str)
    response = model.invoke([
        ("system", "You are a cautious equity research analyst. Write a compact Markdown "
         "research note with sections Snapshot, Financial performance, Valuation, "
         "Bull case, Bear case, and Data gaps. Use ONLY the provided JSON. "
         "Separate reported facts from conditional hypotheses. Include the price date, "
         "currency if known, source, and UTC retrieval time. Null means unavailable; "
         "do not invent company events, competitors, forecasts, filings, catalysts, "
         "price targets, or investment recommendations. Do not label reported "
         "revenue/net income with a fiscal year unless given. The price-return metric "
         "uses the available approximately one-year adjusted-close window, not an "
         "exact calendar year or total shareholder return. Provider metrics may "
         "have mismatched or unspecified reporting periods; disclose this. "
         "Do not suggest Yahoo company-info values are audited."),
        ("human", f"Write the research note for {state['ticker']} using:\n{evidence}"),
    ])
    return {"report": str(response.content)}


def build_graph():
    """The simple original pipeline: fetch -> calculate -> write -> finish."""
    graph = StateGraph(ResearchState)
    graph.add_node("fetch", fetch_data)
    graph.add_node("calculate", calculate)
    graph.add_node("write", write_report)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "calculate")
    graph.add_edge("calculate", "write")
    graph.add_edge("write", END)
    return graph.compile()


def main() -> None:
    """Run the pipeline for one command-line ticker and display its inputs."""
    parser = argparse.ArgumentParser(description="Minimal LangGraph + Groq equity research")
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. AAPL or RELIANCE.NS")
    args = parser.parse_args()
    try:
        result = build_graph().invoke({"ticker": args.ticker.strip().upper()})
        print(result["report"])
        print("\n--- Structured inputs used (verify against primary sources) ---")
        print(json.dumps(result["metrics"], indent=2, allow_nan=False, default=str))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
