# AI Equity Research Platform

Multi-Agent AI research workflow that turns a stock ticker into a deatiled equity research report. It combines a Streamlit interface, a FastAPI backend, a LangGraph analysis pipeline, Yahoo Finance data, and a Groq-hosted language model.

**In plain terms:** choose a company, let the app gather available market and company data, review a generated research note, and inspect the numbers behind it. The tool is for research and learning, not investment advice.

## What this project demonstrates

- **Product thinking:** A user-facing dashboard with ticker search, ten curated company shortcuts, headline metrics, a research note, and downloadable outputs.
- **Backend engineering:** A FastAPI endpoint validates ticker requests and runs the research workflow without exposing the Groq API key to the UI.
- **AI workflow design:** A three-step LangGraph pipeline separates data collection, calculation, and report writing rather than asking an LLM to do everything at once.
- **Responsible AI:** The model is instructed to use supplied data, flag missing information, avoid invented events and price targets, and distinguish facts from hypotheses.
- **Evaluation:** DeepEval scores completed reports for evidence grounding and usefulness, while reporting failed or incomplete runs separately.

## How it works

```text
User selects a company or enters a ticker in Streamlit
                ↓
         FastAPI validates the ticker
                ↓
        LangGraph: fetch → calculate → write
                ↓
  Yahoo Finance data → Python-calculated metrics → Groq research note
                ↓
Streamlit displays the note, metrics, sources, and download options
```

The **fetch** step retrieves roughly one year of adjusted price history and available company information. The **calculate** step performs simple financial arithmetic in Python, so the model is not responsible for computing the displayed ratios. The **write** step sends the structured inputs to Groq to draft the research note. Available Yahoo company-information fields can have different or unspecified reporting periods; the application calls out that limitation rather than treating them as audited financial statements.

This is the *simple sequential pipeline* version of the project, not a parallel multi-agent orchestrator. Calling it an “agentic workflow” here means the research process is implemented as explicit graph steps, not that autonomous specialists search and verify external filings.

## Interface and project files

| File | Role |
|---|---|
| `frontend.py` | Streamlit UI titled **AI Equity Research Platform**. Includes ticker entry, ten featured-company buttons, results, and downloads. |
| `backend.py` | FastAPI service with `/research` and `/health` endpoints. |
| `equity_research_groq.py` | LangGraph research pipeline and CLI entry point. |
| `evaluation.py` | Runs the pipeline on selected tickers and grades the generated notes with DeepEval. |
| `eval_json_to_markdown.py` | Converts evaluation JSON into a Markdown results table. |
| `env.py` | Creates a local `.env` template without overwriting an existing file. |
| `requirements.txt` | Python dependencies for the app; install DeepEval separately if it is not listed. |

The featured companies are **curated examples**, not a real-time ranking of the ten largest companies. Users can also enter other Yahoo Finance tickers, including NSE symbols such as `RELIANCE.NS`.

## Run locally

Use Python 3.10 or newer. From the folder containing the project files:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python env.py
```

Edit the generated `.env` and replace the placeholder with your Groq API key:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Start the backend in one terminal:

```bash
uvicorn backend:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal, with the same virtual environment activated:

```bash
streamlit run frontend.py
```

Streamlit prints the local UI address. The backend's health check is available at `http://127.0.0.1:8000/health`. For a command-line run without the UI, use `python equity_research_groq.py AAPL`. The application needs internet access for Yahoo Finance and Groq. Keep `.env` private and out of version control.

## Evaluation: what the results mean

The included `evaluation.py` uses **DeepEval G-Eval**, an AI-as-a-judge approach. For each ticker, it runs the complete pipeline and then asks a Groq judge to assess (1) whether the note is grounded in the data the pipeline supplied and (2) whether it is useful and appropriately caveated. Each check uses a 0.7 passing threshold by default. Run it with:

```bash
python -m pip install deepeval
python evaluation.py --tickers AAPL MSFT NVDA GOOGL AMZN META TSM AVGO TSLA RELIANCE.NS
python eval_json_to_markdown.py evaluation_results.json --output evaluation_table.md
```

In the supplied ten-ticker run, **eight tickers had both checks completed and scored 1.00 on each**: AAPL, NVDA, AMZN, META, TSM, AVGO, TSLA, and RELIANCE.NS. MSFT received 1.00 for evidence grounding, but its usefulness check did not finish. GOOGL had no completed scores. Both incomplete runs encountered Groq HTTP 429 token-per-minute rate limits; they are **incomplete**, not evidence of poor research quality and not passes. The recorded judge was `Groq/openai/gpt-oss-120b`, and the threshold was 0.7. These are results from the supplied evaluation snapshot, not a claim that every future run will score the same. A practical next step is retry/backoff and pacing between tickers, followed by rerunning the incomplete cases.

> **Important limitation:** The judge compares the report with the Yahoo-derived data supplied to it. A high score does **not** independently verify the source figures against company filings, prove that an investment thesis is correct, or establish performance on a broader benchmark. The current test set is small and contains mostly well-covered companies.
