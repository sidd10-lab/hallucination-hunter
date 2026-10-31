# 🔎 Hallucination Hunter

Ask questions about your PDFs and **check whether the answer is actually true to the document**.
Built with Streamlit + Gemini.

## What makes it different

| Layer | What it does |
|---|---|
| Hybrid retrieval | Gemini embeddings + BM25 keyword search, fused with Reciprocal Rank Fusion, then MMR to avoid duplicate passages |
| Follow-up understanding | "What about its cost?" is rewritten into a standalone search query using the chat |
| Citation checker | Every `[file, p. N]` in the answer is checked against the pages that were really retrieved. Invented pages are flagged |
| Number checker | Any number in the answer that appears nowhere in the evidence is flagged (no AI involved) |
| Quote-verified claim audit | The auditor must quote the PDF for each "SUPPORTED" claim. If the quote isn't in the text, the claim is downgraded |
| PDF page preview | Open the cited page with the verified quote highlighted |
| Stress test | Generates answerable and **trap** questions, runs them, reports hallucination rate and trap-refusal rate |
| Multilingual | Unicode-safe tokenizer (Marathi / Hindi PDFs work); answers follow the question's language |

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env   # then put your real key in .env
python -m streamlit run app.py
```

Run the offline tests (no API key needed): `python -m pip install pytest` then `python -m pytest -q`

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (`app.py`, `engine.py`, `requirements.txt` at the root).
2. share.streamlit.io → **Create app** → branch `main`, file `app.py`.
3. **Advanced settings → Secrets**:

```toml
GEMINI_API_KEY = "your-real-key"
GEN_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"
```

**Never commit `.env` or `.streamlit/secrets.toml`.** (`.streamlit/config.toml` is safe to commit.)

## Files

- `app.py`: Streamlit UI
- `engine.py`: chunking, retrieval, checks, audit (no Streamlit, fully testable)
- `tests/test_engine.py`: 12 offline tests

## Limitations

- Text PDFs only; scanned PDFs need OCR first (the app warns you).
- 40 MB per PDF, 700 combined pages.
- The auditor is also an LLM. The quote check makes it much harder to fool, but always verify important claims on the PDF page.
- Retrieval scores are not truth scores.
- Indexed PDFs are kept in server memory (LRU, never written to disk) and are shared by content hash across sessions on the same server.
