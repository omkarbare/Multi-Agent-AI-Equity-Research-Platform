"""AI Equity Research Platform — Streamlit UI for the FastAPI backend.

Start the backend: uvicorn backend:app --host 127.0.0.1 --port 8000
Start this UI:   streamlit run frontend.py
Override backend endpoint with BACKEND_URL if necessary.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# A curated set of widely followed companies, not a live top-ten market-cap list.
COMPANIES = [
    ("NVDA", "NVIDIA Corporation", "Semiconductors · AI computing"),
    ("AAPL", "Apple Inc.", "Consumer technology · Services"),
    ("GOOGL", "Alphabet Inc.", "Internet services · Advertising"),
    ("MSFT", "Microsoft Corporation", "Enterprise software · Cloud"),
    ("AMZN", "Amazon.com, Inc.", "E-commerce · Cloud"),
    ("TSM", "Taiwan Semiconductor Manufacturing Company Limited", "Semiconductor foundry"),
    ("AVGO", "Broadcom Inc.", "Semiconductors · Infrastructure software"),
    ("META", "Meta Platforms, Inc.", "Social platforms · Advertising"),
    ("TSLA", "Tesla, Inc.", "Electric vehicles · Energy"),
    ("RELIANCE.NS", "Reliance Industries Limited", "Energy · Retail · Telecom"),
]

st.set_page_config(
    page_title="AI Equity Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# An intentional dark-navy + teal system. Streamlit's own light-theme variables
# are overridden too; otherwise buttons and labels can remain white-on-white
# under a user's Streamlit theme. Keep all actionable UI as native widgets.
st.markdown("""
<style>
:root {
  color-scheme: dark;
  --primary-color: #31D0BC;
  --background-color: #0B1423;
  --secondary-background-color: #13243A;
  --text-color: #EAF2F8;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: #0B1423 !important; color: #EAF2F8 !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background: #0B1423 !important; }
.block-container { max-width: 1240px; padding: 2rem 2rem 3.5rem; }
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 { color: #EAF2F8; }
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p { color: #A4B5C8 !important; }
.brand { display: flex; align-items: center; gap: .65rem; color: #31D0BC;
  text-transform: uppercase; letter-spacing: .13em; font-size: .74rem;
  font-weight: 800; padding-bottom: 1.3rem; border-bottom: 1px solid #263D54; }
.brand-mark { width: 23px; height: 23px; display: inline-flex; align-items: center;
  justify-content: center; border: 1px solid #31D0BC; border-radius: 7px;
  font-size: .9rem; line-height: 1; }
.hero { padding: 2.2rem 0 1.5rem; }
.hero h1 { color: #F7FBFD !important; font-size: clamp(2rem, 4vw, 3.5rem);
  line-height: 1.12; margin: 0 0 .7rem; letter-spacing: -.04em; }
.hero p { color: #BACBDC !important; font-size: 1.07rem; line-height: 1.6;
  max-width: 720px; margin: 0; }
.eyebrow { color: #43D8C4 !important; font-size: .76rem; letter-spacing: .14em;
  text-transform: uppercase; font-weight: 800; margin-bottom: .7rem; }
.section-heading { color: #F0F6FB; font-size: 1.25rem; font-weight: 760;
  letter-spacing: -.018em; margin: 1.8rem 0 .3rem; }
.section-copy { color: #A4B5C8 !important; font-size: .89rem; margin-bottom: 1.1rem; }
/* The form is the main workspace and has a distinct, visible surface. */
[data-testid="stForm"] { background: #13243A !important; border: 1px solid #31506B !important;
  border-radius: 16px !important; padding: 1.35rem 1.5rem !important;
  box-shadow: 0 14px 35px rgba(0,0,0,.15); }
[data-testid="stTextInput"] input { background: #091728 !important; color: #F0F7FA !important;
  border: 1px solid #47637C !important; border-radius: 9px !important; }
[data-testid="stTextInput"] input::placeholder { color: #8EA7BA !important; }
[data-testid="stTextInput"] label p { color: #DDEBF4 !important; font-weight: 600; }
[data-testid="stTextInput"] input:focus { border-color: #31D0BC !important;
  box-shadow: 0 0 0 1px #31D0BC !important; }
[data-testid="stFormSubmitButton"] button { min-height: 42px !important;
  background: #31D0BC !important; color: #061927 !important; border: 1px solid #31D0BC !important;
  border-radius: 9px !important; font-weight: 780 !important; }
[data-testid="stFormSubmitButton"] button:hover { background: #71E7D8 !important;
  color: #061927 !important; border-color: #71E7D8 !important; }
/* Secondary company buttons: clear contrast against both page and cards. */
[data-testid="stButton"] button { background: #173048 !important;
  border: 1px solid #3A6279 !important; border-radius: 11px !important;
  color: #F1F7FB !important; min-height: 74px !important;
  white-space: pre-line !important; line-height: 1.5 !important;
  font-size: .86rem !important; font-weight: 640 !important;
  box-shadow: 0 3px 12px rgba(0,0,0,.09); }
[data-testid="stButton"] button p, [data-testid="stButton"] button span {
  color: #F1F7FB !important; white-space: pre-line !important; }
[data-testid="stButton"] button:hover { background: #205069 !important;
  border-color: #31D0BC !important; color: #FFFFFF !important; }
[data-testid="stButton"] button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid #F4C96B !important;
  outline-offset: 2px !important; }
[data-testid="stMetric"] { background: #13243A !important; border: 1px solid #31506B;
  border-radius: 13px; padding: 1.05rem 1.2rem; min-height: 115px; }
[data-testid="stMetricLabel"] p { color: #ADC3D4 !important; }
[data-testid="stMetricValue"] { color: #F5FCFD !important; }
[data-testid="stTabs"] { background: #13243A !important;
  border: 1px solid #31506B; border-radius: 14px; padding: 1rem 1.4rem; }
[data-testid="stTabs"] button { color: #C5D8E4 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #4FDDCA !important; }
[data-testid="stDataFrame"], [data-testid="stJson"] { color: #EAF2F8; }
[data-testid="stDownloadButton"] button { background: #173048 !important;
  color: #EAF2F8 !important; border: 1px solid #4A6B80 !important; }
[data-testid="stDownloadButton"] button:hover { border-color: #31D0BC !important; }
hr { border-color: #31506B !important; }
.footer { color: #9CB1C1; font-size: .81rem; padding-top: .6rem; }
@media(max-width:700px) { .block-container { padding: 1.1rem .85rem 2.5rem; }
  .hero { padding-top: 1.45rem; } .hero h1 { font-size: 2.05rem; } }
</style>
""", unsafe_allow_html=True)


def format_number(value: Any, currency: str | None = None) -> str:
    """Format a reported numeric value; never present missing as zero."""
    if not isinstance(value, (int, float)):
        return "—"
    prefix = f"{currency} " if currency else ""
    for threshold, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= threshold:
            return f"{prefix}{value / threshold:,.2f}{unit}"
    return f"{prefix}{value:,.2f}"


def format_ratio(value: Any, suffix: str = "") -> str:
    """Display an available ratio or an em dash."""
    return f"{value:,.2f}{suffix}" if isinstance(value, (int, float)) else "—"


def request_research(ticker: str) -> dict[str, Any]:
    """Call the local API; the Groq key remains server-side."""
    response = requests.post(f"{API_URL}/research", json={"ticker": ticker}, timeout=(5, 180))
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Research request failed")
        except ValueError:
            detail = "Research request failed"
        raise RuntimeError(f"Backend error ({response.status_code}): {detail}")
    return response.json()


def run_research(ticker: str) -> None:
    """Validate, fetch and remember the latest report through Streamlit reruns."""
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", ticker):
        st.error("Enter a valid Yahoo Finance ticker such as AAPL or RELIANCE.NS.")
        return
    with st.spinner(f"Building a research note for {ticker}…"):
        try:
            st.session_state["research_result"] = request_research(ticker)
        except requests.exceptions.RequestException:
            st.error(f"Could not connect to {API_URL}. Start the FastAPI backend first.")
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))



st.markdown("""
<div class="hero">
  <h1>AI Equity Research Platform</h1>
  <p>Generate a structured, source-aware equity note from market data.
     Review the underlying inputs before you rely on an insight.</p>
</div>
""", unsafe_allow_html=True)

# Primary action first: enter any supported Yahoo Finance ticker.
st.markdown('<div class="section-heading">Start an equity research report</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Enter a listed company ticker to build a report with financial metrics and source details.</div>',
            unsafe_allow_html=True)
with st.form("ticker_form"):
    field, action = st.columns([3.4, 1], vertical_alignment="bottom", gap="medium")
    with field:
        ticker_input = st.text_input("Company ticker", placeholder="AAPL, MSFT, RELIANCE.NS…")
    with action:
        submitted = st.form_submit_button("Generate report", use_container_width=True)

selected_ticker = ticker_input.strip().upper() if submitted else None

# The requested placement: curated buttons appear below the equity research
# entry area rather than above it. They launch the exact same API workflow.
st.markdown('<div class="section-heading">Explore 10 featured companies</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Select a company to run the same report. Curated examples, not a live largest-company ranking.</div>',
            unsafe_allow_html=True)
for row_start in range(0, len(COMPANIES), 2):
    columns = st.columns(2, gap="small")
    for column, (ticker, full_name, business) in zip(columns, COMPANIES[row_start:row_start + 2]):
        with column:
            label = f"{full_name}  ·  {ticker}\n{business}"
            if st.button(label, key=f"featured_{ticker}", use_container_width=True):
                selected_ticker = ticker

if selected_ticker is not None:
    run_research(selected_ticker)

result = st.session_state.get("research_result")
if result:
    metrics = result["metrics"]
    currency = metrics.get("currency")
    st.markdown('<div class="section-heading">Research overview</div>', unsafe_allow_html=True)
    st.caption(f"{metrics.get('company', result['ticker'])} · {result['ticker']}  |  "
               f"Price date: {metrics.get('last_close_date', 'unavailable')}  |  "
               f"Retrieved: {metrics.get('retrieved_utc', 'unavailable')} UTC")
    cards = st.columns(4, gap="small")
    cards[0].metric("Adjusted close", format_number(metrics.get("last_adjusted_close"), currency))
    cards[1].metric("Window return", format_ratio(metrics.get("price_return_available_window_pct"), "%"))
    cards[2].metric("Market cap", format_number(metrics.get("market_cap"), currency))
    cards[3].metric("Trailing P/E", format_ratio(metrics.get("trailing_pe")))

    note, indicators, inputs = st.tabs(["Research note", "Financial indicators", "Sources & inputs"])
    with note:
        st.markdown(result["report"])
        st.download_button("Download note (.md)", result["report"],
                           file_name=f"{result['ticker']}_research.md", mime="text/markdown")
    with indicators:
        rows = [
            ("Reported revenue", format_number(metrics.get("revenue_reported_by_provider"), currency)),
            ("Reported net income", format_number(metrics.get("net_income_reported_by_provider"), currency)),
            ("Reported profit margin", format_ratio(metrics.get("profit_margin_pct_reported_by_provider"), "%")),
            ("Reported revenue growth", format_ratio(metrics.get("revenue_growth_pct_reported_by_provider"), "%")),
            ("Reported earnings growth", format_ratio(metrics.get("earnings_growth_pct_reported_by_provider"), "%")),
            ("Reported free cash flow", format_number(metrics.get("free_cash_flow_reported_by_provider"), currency)),
            ("Net debt from provider fields", format_number(metrics.get("net_debt_from_provider_fields"), currency)),
            ("Forward P/E", format_ratio(metrics.get("forward_pe"))),
            ("Price / book", format_ratio(metrics.get("price_to_book"))),
        ]
        st.dataframe([{"Indicator": label, "Value": value} for label, value in rows],
                     hide_index=True, use_container_width=True)
        st.caption("Provider metrics may have different or unspecified reporting periods.")
    with inputs:
        st.write("Yahoo Finance source:", metrics.get("source", "Unavailable"))
        st.write("Retrieved (UTC):", metrics.get("retrieved_utc", "Unavailable"))
        if metrics.get("warning"):
            st.warning(metrics["warning"])
        st.json(metrics)
        st.download_button("Download inputs (.json)", json.dumps(metrics, indent=2, default=str),
                           file_name=f"{result['ticker']}_inputs.json", mime="application/json")

st.divider()
st.markdown('<div class="footer">Research support only. Verify figures against primary filings. Not investment advice.</div>',
            unsafe_allow_html=True)
