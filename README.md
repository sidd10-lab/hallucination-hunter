# 🔎 Hallucination Hunter

Ask questions about your PDFs and **check whether each answer is actually true to the document**.
Streamlit + Gemini, with an optional free Supabase library.

## Features

| | |
|---|---|
| 🛡 **Trust score** | One 0–100 number per answer: claim faithfulness, minus penalties for fake citations and ungrounded numbers |
| 🖍 **Inline highlighting** | Answer sentences colored green / red / yellow by audit verdict; hover for the reason |
| 🛠 **Self-repair** | If the audit finds unsupported claims, the answer is rewritten, re-audited, and shown before vs after |
| 📷 **OCR for scanned PDFs** | Gemini vision reads image-only pages (Marathi / Hindi too) |
| ☁️ **Saved library** | Indexed PDFs are stored in a free Supabase project, so you never re-embed them |
| 🎯 **Quote-verified audit** | "Supported" needs a quote that really exists in the PDF text, otherwise it is downgraded |
| 🔎 **Hybrid retrieval** | Gemini embeddings + BM25, fused with RRF, then MMR for variety; follow-ups are rewritten |
| ✅ **Deterministic checks** | Every `[file, p. N]` citation and every number is checked without any AI |
| 🧪 **Stress test** | Answerable + trap questions, reports hallucination rate |

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env   # then put your keys in .env
python -m streamlit run app.py
```

Offline tests (no keys needed): `python -m pip install pytest` then `python -m pytest -q`

## Deploy on Streamlit Community Cloud

Push to GitHub, create the app (main file `app.py`), and put this in **Advanced settings → Secrets**:

```toml
GEMINI_API_KEY = "your-gemini-key"
GEN_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"

# optional, for the saved library
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-service-role-key"
```

## Free cloud library (Supabase)

1. Create a free project at supabase.com.
2. **SQL Editor → New query**, paste `supabase_schema.sql`, click **Run**.
3. **Project Settings → API**: copy the **Project URL** and the **service_role** key.
4. Add them as `SUPABASE_URL` and `SUPABASE_KEY` in Secrets, then reboot the app.
5. In the app sidebar, enter a **workspace code** (6+ characters, keep it secret). Your PDFs are saved under it.

The service_role key bypasses all database rules. Keep it **only** in Secrets or `.env`, never in the repo.
The workspace code is a simple passphrase, not full account security: anyone who knows it can open that library.

## Files

- `app.py`: UI
- `engine.py`: chunking, OCR, retrieval, checks, audit, repair, highlighting
- `store.py`: Supabase client
- `supabase_schema.sql`: run once in Supabase
- `tests/`: 21 offline tests

## Limitations

- 40 MB per PDF, 700 combined pages, OCR up to 40 pages per PDF per upload (uses extra Gemini calls).
- The auditor is also an LLM. The quote check makes it much harder to fool, but verify important claims on the cited page.
- OCR can misread poor scans; answers inherit those errors.
- Free Gemini and Supabase tiers have quotas (Supabase free: 500 MB database, 1 GB storage).
