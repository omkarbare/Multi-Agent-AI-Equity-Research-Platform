"""App for streamlit cloud deployment.

Local: pip install -r requirements.txt; python env.py; edit .env;
       streamlit run streamlit_app.py

There is NO FastAPI dependency: the UI invokes the existing LangGraph pipeline
in-process. Keep equity_research_groq.py next to this file.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import streamlit as st

from equity_research_groq import build_graph

COMPANIES = [
    ("NVDA", "NVIDIA Corporation", "Semiconductors · AI computing"),
    ("AAPL", "Apple Inc.", "Consumer technology · Services"),
    ("GOOGL", "Alphabet Inc.", "Internet services · Advertising"),
    ("MSFT", "Microsoft Corporation", "Enterprise software · Cloud"),
    ("AMZN", "Amazon.com, Inc.", "E-commerce · Cloud"),
    ("TSM", "Taiwan Semiconductor Manufacturing Company Limited", "Semiconductor foundry"),
]

st.set_page_config(page_title="AI Equity Research Platform", page_icon="📊", layout="wide")

# Explicit dark surfaces and high-contrast controls to avoid white-on-white
# appearance even when Streamlit has a different theme configured.
st.markdown("""
<style>
:root { color-scheme: dark; --primary-color:#31D0BC; --background-color:#0B1423;
        --secondary-background-color:#13243A; --text-color:#EAF2F8; }
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
 background:#0B1423!important;color:#EAF2F8!important; }
[data-testid="stHeader"] { background:#0B1423!important; }
.block-container { max-width:1200px;padding:2rem 2rem 3rem; }
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label { color:#EAF2F8; }
[data-testid="stCaptionContainer"] p { color:#B3C8D7!important; }
.brand { font-size:.75rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;
 color:#31D0BC;padding-bottom:1.1rem;border-bottom:1px solid #31506B; }
.hero { padding:2.15rem 0 1.25rem; }
.hero h1 { font-size:clamp(2.1rem,4vw,3.35rem);line-height:1.13;letter-spacing:-.04em;
 color:#F7FBFD!important;margin:0 0 .6rem; }
.hero p { color:#B6C9D8!important;max-width:710px;font-size:1.05rem;line-height:1.55; }
.eyebrow { color:#4CE4D0!important;font-size:.77rem;letter-spacing:.14em;
 text-transform:uppercase;font-weight:800;margin-bottom:.65rem; }
.section { font-size:1.22rem;font-weight:750;margin:1.75rem 0 .35rem;color:#F0F8FB; }
.muted { color:#AEC1D0!important;font-size:.89rem;margin-bottom:1.05rem; }
[data-testid="stForm"] { background:#13243A!important;border:1px solid #31506B!important;
 border-radius:16px!important;padding:1.35rem!important; }
[data-testid="stTextInput"] input { background:#0B1C2C!important;color:#F1FAFD!important;
 border:1px solid #4F7087!important;border-radius:9px!important; }
[data-testid="stTextInput"] input::placeholder { color:#9CB3C3!important; }
[data-testid="stTextInput"] label p { color:#EAF2F8!important; }
[data-testid="stFormSubmitButton"] button { background:#31D0BC!important;color:#061927!important;
 border:1px solid #31D0BC!important;min-height:43px;border-radius:9px;font-weight:750; }
[data-testid="stFormSubmitButton"] button:hover { background:#70E7D8!important;color:#061927!important; }
[data-testid="stButton"] button { background:#173048!important;color:#F1F7FB!important;
 border:1px solid #466B81!important;border-radius:10px!important;min-height:75px!important;
 white-space:pre-line!important;line-height:1.5!important;font-weight:630!important; }
[data-testid="stButton"] button p,[data-testid="stButton"] button span { color:#F1F7FB!important;
 white-space:pre-line!important; }
[data-testid="stButton"] button:hover { background:#23516A!important;border-color:#31D0BC!important; }
[data-testid="stButton"] button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible { outline:3px solid #F4C96B!important; }
[data-testid="stMetric"] { background:#13243A!important;border:1px solid #31506B;
 border-radius:13px;padding:1rem;min-height:110px; }
[data-testid="stMetricLabel"] p { color:#B6C9D8!important; }
[data-testid="stMetricValue"] { color:#FFFFFF!important; }
[data-testid="stTabs"] { background:#13243A!important;border:1px solid #31506B;
 border-radius:13px;padding:1rem 1.35rem; }
[data-testid="stTabs"] button { color:#D0E2EC!important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#49E0CD!important; }
[data-testid="stDownloadButton"] button { background:#173048!important;color:#F3F9FB!important;
 border:1px solid #466B81!important; }
hr { border-color:#31506B!important; }
@media(max-width:700px){.block-container{padding:1rem .9rem 2rem}.hero h1{font-size:2rem}}
</style>
""", unsafe_allow_html=True)


def configure_groq_key() -> bool:
    """Use Community Cloud secrets, falling back to a local .env-based key.

    The existing pipeline reads GROQ_API_KEY using os.getenv(). Setting the
    process environment here lets the same file run locally or on Cloud.
    """
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
        secret_model = st.secrets.get("GROQ_MODEL")
    except (FileNotFoundError, KeyError):
        secret_key = secret_model = None
    if secret_key:
        os.environ["GROQ_API_KEY"] = str(secret_key)
    if secret_model:
        os.environ["GROQ_MODEL"] = str(secret_model)
    return bool(os.getenv("GROQ_API_KEY")) and os.getenv("GROQ_API_KEY") != "replace_with_your_groq_api_key"


@st.cache_resource
def get_graph():
    """Reuse the compiled graph on reruns; do not cache market-data responses."""
    return build_graph()


def compact(value: Any, currency: str | None = None) -> str:
    """Format a number compactly, preserving missing values as unknown."""
    if not isinstance(value, (int, float)):
        return "—"
    prefix = f"{currency} " if currency else ""
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"{prefix}{value / size:,.2f}{suffix}"
    return f"{prefix}{value:,.2f}"


def ratio(value: Any, suffix: str = "") -> str:
    """Format a numeric ratio while keeping missing distinct from zero."""
    return f"{value:,.2f}{suffix}" if isinstance(value, (int, float)) else "—"


def run_research(ticker: str) -> None:
    """Run the real research graph directly inside the Streamlit process."""
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", ticker):
        st.error("Enter a valid ticker, such as AAPL or RELIANCE.NS.")
        return
    if not configure_groq_key():
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Cloud app secrets, or local .env.")
        return
    try:
        with st.spinner(f"Researching {ticker} — retrieving data and writing the note…"):
            result = get_graph().invoke({"ticker": ticker})
        st.session_state["research_result"] = result
    except Exception as exc:
        # Do not print exception strings that might include upstream request data.
        import logging
        logging.getLogger(__name__).exception("Research failed for %s", ticker)
        st.error("Research could not be completed. Check the app logs for details and try again.")


st.markdown("""
<div class="hero">
<div class="eyebrow">Created by Omkar Bare | Welcome to my portfolio project</div>
<h1>AI Equity Research Platform</h1>
<p>Generate a source-aware equity research report & inspect the figures used by AI to write it.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section">Start an equity research report</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Enter any supported Yahoo Finance symbol. Your report appears below the company choices.</div>',
            unsafe_allow_html=True)
with st.form("ticker_form"):
    field, action = st.columns([3.4, 1], vertical_alignment="bottom", gap="medium")
    with field:
        ticker_input = st.text_input("Company ticker", placeholder="AAPL, MSFT, RELIANCE.NS…")
    with action:
        submitted = st.form_submit_button("Generate report", use_container_width=True)
selected = ticker_input.strip().upper() if submitted else None

st.markdown('<div class="section">Explore 10 featured companies</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Curated examples, not a live market-cap ranking. Click to generate a report.</div>',
            unsafe_allow_html=True)
for start in range(0, len(COMPANIES), 2):
    columns = st.columns(2, gap="small")
    for column, (symbol, name, business) in zip(columns, COMPANIES[start:start + 2]):
        with column:
            if st.button(f"{name} · {symbol}\n{business}", key=f"featured_{symbol}", use_container_width=True):
                selected = symbol

if selected is not None:
    run_research(selected)

result = st.session_state.get("research_result")
if result:
    m = result["metrics"]
    currency = m.get("currency")
    ticker = result["ticker"]
    st.markdown('<div class="section">Research overview</div>', unsafe_allow_html=True)
    st.caption(f"{m.get('company', ticker)} · {ticker} | Price: {m.get('last_close_date', 'unavailable')} | "
               f"Retrieved: {m.get('retrieved_utc', 'unavailable')} UTC")
    cards = st.columns(4, gap="small")
    cards[0].metric("Adjusted close", compact(m.get("last_adjusted_close"), currency))
    cards[1].metric("Window return", ratio(m.get("price_return_available_window_pct"), "%"))
    cards[2].metric("Market cap", compact(m.get("market_cap"), currency))
    cards[3].metric("Trailing P/E", ratio(m.get("trailing_pe")))

    note, indicators, evidence = st.tabs(["Research note", "Financial indicators", "Sources & inputs"])
    with note:
        st.markdown(result["report"])
        st.download_button("Download research note (.md)", result["report"],
                           file_name=f"{ticker}_research.md", mime="text/markdown")
    with indicators:
        rows = [
            ("Reported revenue", compact(m.get("revenue_reported_by_provider"), currency)),
            ("Reported net income", compact(m.get("net_income_reported_by_provider"), currency)),
            ("Profit margin", ratio(m.get("profit_margin_pct_reported_by_provider"), "%")),
            ("Revenue growth", ratio(m.get("revenue_growth_pct_reported_by_provider"), "%")),
            ("Earnings growth", ratio(m.get("earnings_growth_pct_reported_by_provider"), "%")),
            ("Free cash flow", compact(m.get("free_cash_flow_reported_by_provider"), currency)),
            ("Net debt", compact(m.get("net_debt_from_provider_fields"), currency)),
            ("Forward P/E", ratio(m.get("forward_pe"))),
            ("Price / book", ratio(m.get("price_to_book"))),
        ]
        st.dataframe([{"Indicator": label, "Value": value} for label, value in rows],
                     hide_index=True, use_container_width=True)
        st.caption("Provider fields may refer to different or unspecified reporting periods.")
    with evidence:
        st.write("Source:", m.get("source", "Unavailable"))
        st.write("Retrieved (UTC):", m.get("retrieved_utc", "Unavailable"))
        if m.get("warning"):
            st.warning(m["warning"])
        st.json(m)
        st.download_button("Download inputs (.json)", json.dumps(m, indent=2, default=str),
                           file_name=f"{ticker}_inputs.json", mime="application/json")

st.divider()
st.caption("Research support only · Check figures against company filings · Not investment advice")
