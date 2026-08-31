# MediGuide AI

An educational Streamlit + LangChain prototype that collects basic patient
information and symptoms and returns structured, safety-first guidance from
an OpenAI chat model.

> **This is an educational prototype only.** It is not a medical device, not
> a licensed doctor, and must never be used for real diagnosis or treatment.
> If you or someone else may be experiencing a medical emergency, call your
> local emergency number or go to the nearest emergency room immediately.

## Features

- **API key gate page**: on first launch, the user pastes their own OpenAI
  API key (kept only in the browser session, never written to disk). If a
  key is already set via `.env`, this page is skipped automatically.
- Patient intake form: age, gender, symptoms (multiselect + free text),
  duration, severity slider, existing conditions, medications, notes.
- `ChatOpenAI` model integration via `langchain-openai`.
- A single-string `PromptTemplate` **and** a `ChatPromptTemplate`
  (System + Human messages) both used to build the structured-JSON request.
- A standalone `SystemMessage` / `HumanMessage` / `AIMessage` demo showing
  how raw chat messages and conversation history work in LangChain.
- Structured JSON output (summary, possible conditions, urgency level, next
  steps, doctor questions, warning signs) parsed defensively so malformed
  output never crashes the app.
- A reusable `LLMChain` for the assessment.
- Live streaming of a human-readable narrative via `.stream()` and
  `st.write_stream()`.
- Switchable **InMemoryCache** and **SQLiteCache**, registered globally with
  `set_llm_cache(...)`.
- A results dashboard using `st.metric`, `st.warning`, `st.info`, `st.error`,
  `st.success`, `st.expander`, tabs, and columns.
- Prominent medical disclaimers on every screen.

## Project structure

```
medical_ai_assistant/
├── app.py                  # Streamlit UI (run this)
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py            # settings + form options
│   ├── prompts.py            # PromptTemplate + ChatPromptTemplate + JSON schema
│   ├── chains.py              # ChatOpenAI, LLMChain, streaming, raw message demo
│   ├── cache_manager.py        # in-memory + SQLite caching switches
│   └── utils.py                # safe JSON parsing + helpers
└── docs/
    └── Medical_AI_Assignment.pdf
```

**Data flow:** form input → build prompt inputs → select cache → build LLM +
`LLMChain` → run chain → parse JSON safely → stream narrative → render
dashboard.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional, for local dev) Copy the example env file and add your key:
   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY=sk-...
   ```
   If you skip this step, the app will simply show the "enter your API key"
   page on first launch instead.
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Never commit your real `.env` file — it is already listed in
   `.gitignore` (create one if it doesn't exist, containing at least `.env`).

## Caching: in-memory vs. SQLite

| | InMemoryCache | SQLiteCache |
|---|---|---|
| Storage | RAM | A file on disk (`.db`) |
| Speed | Fastest | Fast, slightly slower |
| Survives app restart? | No | Yes |
| Best for | One session | Reusing across sessions / restarts |

Both caches key off the *exact* prompt text sent to the model. `set_llm_cache(...)`
registers the chosen cache globally, and LangChain checks it automatically
before every model call — so submitting the same form twice with a cache
enabled is visibly faster the second time, with an identical result. Choose
the cache type from the sidebar; "No caching" disables this behaviour so
every submission calls the API fresh.

## Testing scenarios

| # | Input | Expected behaviour |
|---|---|---|
| 1 | Age 25, runny nose + sore throat, 1-3 days, severity 2 | Urgency LOW; calm monitoring advice |
| 2 | Age 40, fever + cough, 4-7 days, severity 6 | Urgency MEDIUM/HIGH; advises seeing a professional |
| 3 | Severe chest pain + shortness of breath | Urgency HIGH/EMERGENCY; urges immediate help |
| 4 | Submit the same form twice (cache on) | Second run is faster; identical result |
| 5 | Empty symptoms | App warns the user and does not call the API |
| 6 | Language = Urdu | Guidance text returns in Urdu |

## Notes

- This project is for education only. It is not a medical device and must
  not be used for real diagnosis or treatment.
- The API key you enter is used directly to call OpenAI's API from your own
  browser session and is not persisted anywhere by this app.
