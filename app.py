from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

import engine as E

load_dotenv()
APP_NAME = "Hallucination Hunter"
MAX_FILE_MB, MAX_PAGES = 40, 700
VIEWS = ["💬 Chat", "📝 Summaries", "🧪 Stress test", "📊 Analytics"]

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        value = ""
    return str(value or os.getenv(name, default)).strip()


API_KEY = secret("GEMINI_API_KEY")
GEN_MODEL = secret("GEN_MODEL", "gemini-3.6-flash")
EMBED_MODEL = secret("EMBED_MODEL", "gemini-embedding-001")

st.markdown("""
<style>
.hero{padding:1.3rem 1.5rem;border:1px solid #2d4166;border-radius:18px;margin-bottom:1rem;
background:radial-gradient(circle at 85% 15%,#34265f,#1a2848 45%,#14213a)}
.hero h1{margin:.1rem 0 .2rem 0;padding:0}.eyebrow{color:#9db8ef;text-transform:uppercase;letter-spacing:.16em;font-size:.7rem;font-weight:700}
.muted{color:#a9b8d4}
div[data-testid="stMetric"]{background:#121e34;border:1px solid #293d60;padding:.7rem .9rem;border-radius:14px}
</style>
<div class="hero"><div class="eyebrow">Document-grounded AI reliability</div>
<h1>🔎 Hallucination Hunter</h1>
<div class="muted">Ask your PDFs, verify every citation, and audit each claim against the source text.</div></div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("Gemini API key is missing. Add GEMINI_API_KEY to your local .env or Streamlit Cloud Secrets.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llm(key: str, gen_model: str, embed_model: str) -> E.GeminiLLM:
    return E.GeminiLLM(key, gen_model, embed_model)


llm = get_llm(API_KEY, GEN_MODEL, EMBED_MODEL)


class DocStore:
    """Shared in-memory LRU of indexed PDFs, keyed by content hash (nothing is written to disk)."""

    def __init__(self, max_docs: int = 8, max_bytes: int = 160 * 2**20):
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.max_docs, self.max_bytes = max_docs, max_bytes

    def get(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key, doc):
        with self.lock:
            self.items[key] = doc
            self.items.move_to_end(key)
            while len(self.items) > 1 and (len(self.items) > self.max_docs
                                           or sum(len(d["data"]) for d in self.items.values()) > self.max_bytes):
                self.items.popitem(last=False)


@st.cache_resource
def doc_store() -> DocStore:
    return DocStore()


@st.cache_resource(max_entries=4, show_spinner=False)
def get_corpus(sig_key: str, _docs: list) -> E.Corpus:  # noqa: ARG001 - sig_key is the cache key
    return E.Corpus(_docs)


@st.cache_data(show_spinner=False, max_entries=64)
def page_png(digest: str, page: int, highlights: tuple, _data: bytes) -> bytes:  # noqa: ARG001
    return E.render_page(_data, page, highlights)


def explain(e: Exception) -> str:
    m = str(e).upper()
    if "RESOURCE_EXHAUSTED" in m or "429" in m:
        return "Gemini quota / rate limit reached. Wait a minute or check your API usage."
    if "503" in m or "UNAVAILABLE" in m or "OVERLOADED" in m:
        return "Gemini is temporarily busy. Wait 10–20 seconds and retry."
    if "API_KEY" in m or "PERMISSION_DENIED" in m or "401" in m or "403" in m:
        return "The API key looks invalid or lacks access. Check GEMINI_API_KEY."
    if "404" in m or "NOT_FOUND" in m:
        return "Model not found. Check GEN_MODEL / EMBED_MODEL in .env or Secrets."
    return ""


# --------------------------------------------------------------------------- state
for k, v in {"messages": [], "pending": None, "summaries": {}, "stress": None, "suggestions": None,
             "signature": None, "view": VIEWS[0]}.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 📚 Knowledge base")
    uploads = st.file_uploader("Upload one or more text-based PDFs", type=["pdf"], accept_multiple_files=True)
    scope_box = st.container()
    with st.expander("⚙️ Settings"):
        top_k = st.slider("Evidence passages", 3, 12, 6)
        chunk_size = st.slider("Passage size (words)", 120, 400, 220, step=20)
        overlap = st.slider("Passage overlap (words)", 0, 100, 40, step=10)
        audit_on = st.toggle("Audit every answer (recommended)", True,
                             help="Extra Gemini call that checks each claim and verifies its quote against the PDF text.")
        rewrite_on = st.toggle("Understand follow-up questions", True,
                               help="Rewrites 'what about its cost?' into a standalone search query using the chat.")
        mmr_on = st.toggle("Diverse evidence (MMR)", True, help="Avoids returning near-duplicate passages.")
        min_rel = st.slider("Abstain below relevance", 0.0, 0.8, 0.0, step=0.05,
                            help="If even the best passage is less similar than this, skip the model and abstain. 0 = off.")
        st.caption(f"Answer model: {llm.active}\nEmbeddings: {EMBED_MODEL}")
    if st.button("Clear chat and results", width="stretch"):
        st.session_state.update(messages=[], suggestions=None, stress=None, summaries={}, pending=None)
        st.rerun()
    st.caption("Keep API keys in .env or Streamlit Secrets. Never commit them.")

if not uploads:
    st.info("Upload one or more PDFs in the left panel to begin.")
    st.markdown("**What you get:** hybrid BM25 + semantic retrieval · citation checker · number checker · "
                "quote-verified claim audit · page preview with highlights · hallucination stress test.")
    st.stop()

# --------------------------------------------------------------------------- load documents
store = doc_store()
loaded, names_seen, total_pages = [], set(), 0
for up in uploads:
    name = up.name
    while name in names_seen:
        name = name.replace(".pdf", "") + " (2).pdf"
    names_seen.add(name)
    data = up.getvalue()
    if len(data) > MAX_FILE_MB * 2**20:
        st.error(f"{name} exceeds {MAX_FILE_MB} MB.")
        continue
    digest = hashlib.sha256(data).hexdigest()
    key = (name, digest, chunk_size, overlap, EMBED_MODEL)
    doc = store.get(key)
    if doc is None:
        bar = st.progress(0.0, text=f"Indexing {name}…")
        try:
            pages = E.page_count(data)
            if total_pages + pages > MAX_PAGES:
                bar.empty()
                st.error(f"Combined page limit is {MAX_PAGES}; {name} would exceed it.")
                continue
            doc = E.build_doc(llm, name, digest, data, chunk_size, overlap,
                              lambda f, n=name: bar.progress(f, text=f"Embedding {n}… {int(f * 100)}%"))
            store.put(key, doc)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not index {name}: {e}")
            continue
        finally:
            bar.empty()
    total_pages += doc["pages"]
    loaded.append(doc)

if not loaded:
    st.stop()
for d in loaded:
    if d["text_pages"] < 0.6 * d["pages"]:
        st.warning(f"{d['name']}: only {d['text_pages']} of {d['pages']} pages have selectable text. "
                   "Scanned pages need OCR first, so answers may miss content.")

with scope_box:
    chosen = st.multiselect("Search in", [d["name"] for d in loaded], default=[d["name"] for d in loaded])
docs = [d for d in loaded if d["name"] in chosen]
if not docs:
    st.warning("Select at least one document to search.")
    st.stop()
by_name = {d["name"]: d for d in docs}

sig = tuple((d["name"], d["digest"]) for d in docs) + (chunk_size, overlap)
if st.session_state.signature != sig:  # documents or chunking changed -> old answers no longer apply
    st.session_state.signature = sig
    st.session_state.update(messages=[], suggestions=None, stress=None, summaries={}, pending=None)
corpus = get_corpus(json.dumps(sig), docs)

c1, c2, c3, c4 = st.columns(4)
c1.metric("PDF files", len(docs))
c2.metric("Pages", sum(d["pages"] for d in docs))
c3.metric("Passages", len(corpus.chunks))
c4.metric("Words", f"{sum(d['words'] for d in docs):,}")

# chat_input must live at the top level so Streamlit pins it to the bottom of the page
typed = st.chat_input("Ask a question about your PDFs…")
if typed and typed.strip():
    st.session_state.pending = typed.strip()
    st.session_state.view = VIEWS[0]
view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or VIEWS[0]


# --------------------------------------------------------------------------- rendering helpers
def chat_context() -> list[dict]:
    return [{"role": "User" if m["role"] == "user" else "Assistant", "content": m["content"]}
            for m in st.session_state.messages[-6:]]


def render_result(res: dict, key: str) -> None:
    a, cit = res.get("audit"), res["citations"]
    if res["abstained"]:
        st.caption("🟦 The model abstained: the retrieved passages did not contain the answer.")
    else:
        total_c = cit["valid"] + cit["invalid"]
        if total_c == 0:
            st.warning("⚠️ The answer has no page citations, so it can't be traced to the PDF.")
        elif cit["invalid"]:
            bad = ", ".join(f"{i['file']} p. {i['page']}" for i in cit["items"] if not i["valid"])
            st.warning(f"⚠️ {cit['invalid']} of {total_c} citations point to pages that were not in the evidence: {bad}")
        else:
            st.caption(f"✅ All {total_c} citations match the retrieved pages.")
        if res["numbers"]:
            st.warning("⚠️ Numbers in the answer that appear nowhere in the evidence: " + ", ".join(res["numbers"]))
    if a:
        x, y, z, w = st.columns(4)
        x.metric("Supported", a["counts"]["SUPPORTED"])
        y.metric("Contradicted", a["counts"]["CONTRADICTED"])
        z.metric("Unverifiable", a["counts"]["UNVERIFIABLE"])
        w.metric("Faithfulness", f"{a['faithfulness']}%" if a["faithfulness"] is not None else "N/A")
        if a["downgraded"]:
            st.caption(f"🔎 {a['downgraded']} claim(s) were downgraded because the judge's quote was not found in the PDF text.")
        if a["claims"]:
            with st.expander("🔬 Claim-by-claim audit", expanded=a["counts"]["SUPPORTED"] < a["total"]):
                st.write(a["summary"])
                df = pd.DataFrame(a["claims"])[["claim", "verdict", "file", "page", "quote", "reason"]]
                st.dataframe(df, width="stretch", hide_index=True)
    elif res.get("audit_error"):
        st.warning("Answer succeeded, but the audit failed: " + res["audit_error"])
    marks: dict = {}
    for cl in (a or {}).get("claims", []):
        if cl["quote_verified"]:
            marks.setdefault((cl["file"], cl["page"]), []).append(cl["quote"])
    with st.expander(f"📚 Evidence ({len(res['sources'])} passages) · {res['latency']}s · {res['model']}"):
        st.caption("Similarity is not a truth score. Read the passage or open the page to check.")
        for i, s in enumerate(res["sources"], 1):
            st.markdown(f"**{i}. {s['file']} · page {s['page']}** "
                        f"<span class='muted'>semantic {s['semantic']:.2f} · keyword {s['keyword']:.2f}</span>",
                        unsafe_allow_html=True)
            st.write(s["text"])
            d = by_name.get(s["file"])
            if d and st.toggle("Show PDF page", key=f"{key}_pg{i}"):
                st.image(page_png(d["digest"], s["page"], tuple(marks.get((s["file"], s["page"]), [])[:3]), d["data"]),
                         width="stretch")
            st.divider()


def run_question(q: str) -> None:
    t0 = time.time()
    chat = chat_context() if rewrite_on else []
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                query, sources, top = E.prepare(llm, corpus, q, chat, top_k, rewrite_on, mmr_on)
            if query != q:
                st.caption(f"🔁 Searched for: {query}")
            if top < min_rel:
                answer = E.ABSTAIN
                st.markdown(answer)
            else:
                answer = st.write_stream(llm.stream(E.answer_prompt(q, sources, chat)))
            with st.spinner("Checking citations and auditing claims…" if audit_on else "Checking citations…"):
                res = E.finalize(llm, q, query, answer, sources, top, audit_on, t0)
            aid = len(st.session_state.messages) + 1
            st.session_state.messages += [{"role": "user", "content": q},
                                          {"role": "assistant", "content": answer, "result": res}]
            render_result(res, f"m{aid}")
        except Exception as e:  # noqa: BLE001
            st.error(f"That question failed: {e}")
            hint = explain(e)
            if hint:
                st.info(hint)


# --------------------------------------------------------------------------- views
if view == VIEWS[0]:
    msgs = st.session_state.messages
    if not msgs and not st.session_state.pending:
        st.markdown("##### Try it")
        if st.button("💡 Suggest questions from my documents"):
            try:
                with st.spinner("Reading your documents…"):
                    st.session_state.suggestions = E.suggest_questions(llm, corpus, 4, 0)["answerable"]
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not generate suggestions: {e}")
        for i, s in enumerate(st.session_state.suggestions or []):
            if st.button(s, key=f"sug{i}", width="stretch"):
                st.session_state.pending = s
                st.rerun()
    for i, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("result"):
                render_result(m["result"], f"m{i}")
    if st.session_state.pending:
        q, st.session_state.pending = st.session_state.pending, None
        run_question(q)

elif view == VIEWS[1]:
    st.subheader("Document summaries")
    st.caption("Built from evenly spaced passages across each PDF (not the whole text), so treat it as an overview.")
    if st.button("✨ Generate summaries", type="primary"):
        for d in docs:
            try:
                with st.spinner(f"Summarizing {d['name']}…"):
                    st.session_state.summaries[d["name"]] = E.summarize_doc(llm, d)
            except Exception as e:  # noqa: BLE001
                st.error(f"{d['name']}: summary failed: {e}")
    shown = {n: v for n, v in st.session_state.summaries.items() if n in by_name}
    for n, (text, used) in shown.items():
        with st.expander(f"📄 {n}", expanded=True):
            st.caption(f"Based on {used} of {len(by_name[n]['chunks'])} passages")
            st.markdown(text)
    if shown:
        md = "\n\n".join(f"# {n}\n\n{t}" for n, (t, _) in shown.items())
        st.download_button("Download summaries", md, "hallucination_hunter_summaries.md", "text/markdown")

elif view == VIEWS[2]:
    st.subheader("Hallucination stress test")
    st.write("Generates questions your PDFs **can** answer and **trap** questions they can't, runs them through the full "
             "pipeline, and reports how often the model makes things up.")
    cA, cB = st.columns(2)
    n_ans = cA.slider("Answerable questions", 2, 8, 4)
    n_trap = cB.slider("Trap questions", 2, 8, 4)
    st.caption(f"Uses about {1 + (n_ans + n_trap) * 3} Gemini calls. Trap questions are generated from a sample of the "
               "text, so one may turn out answerable; the report flags that case.")
    if st.button("🧪 Run stress test", type="primary"):
        try:
            with st.spinner("Generating questions…"):
                qs = E.suggest_questions(llm, corpus, n_ans, n_trap)
            jobs = [("answerable", q) for q in qs["answerable"]] + [("trap", q) for q in qs["traps"]]
            rows, bar = [], st.progress(0.0, text="Running…")
            for i, (kind, q) in enumerate(jobs, 1):
                bar.progress(i / len(jobs), text=f"{i}/{len(jobs)}: {q[:70]}")
                try:
                    r = E.answer_once(llm, corpus, q, k=top_k, mmr=mmr_on, audit_on=True, min_rel=min_rel)
                    a = r.get("audit") or {}
                    rows.append({"Type": kind, "Question": q, "Outcome": E.classify(kind, r),
                                 "Faithfulness %": a.get("faithfulness"), "Answer": r["answer"][:300]})
                except Exception as e:  # noqa: BLE001
                    rows.append({"Type": kind, "Question": q, "Outcome": f"Error: {str(e)[:80]}",
                                 "Faithfulness %": None, "Answer": ""})
            bar.empty()
            st.session_state.stress = rows
        except Exception as e:  # noqa: BLE001
            st.error(f"Stress test failed: {e}")
    rows = st.session_state.stress
    if rows:
        df = pd.DataFrame(rows)
        answered = df[~df["Outcome"].isin(["Correctly abstained", "Wrongly abstained"]) & ~df["Outcome"].str.startswith("Error")]
        bad = int((answered["Outcome"] == "Unsupported claims").sum())
        traps = df[df["Type"] == "trap"]
        ok_traps = int((traps["Outcome"] == "Correctly abstained").sum())
        x, y, z = st.columns(3)
        x.metric("Hallucination rate", f"{round(100 * bad / len(answered))}%" if len(answered) else "N/A",
                 help="Answers containing at least one contradicted or unverifiable claim.")
        y.metric("Trap questions refused", f"{ok_traps}/{len(traps)}")
        z.metric("Avg faithfulness", f"{df['Faithfulness %'].dropna().mean():.0f}%" if df["Faithfulness %"].notna().any() else "N/A")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("⬇️ Download stress test CSV", df.to_csv(index=False).encode(), "stress_test.csv", "text/csv")

else:
    st.subheader("Analytics & export")
    results = [m["result"] for m in st.session_state.messages if m.get("result")]
    if not results:
        st.info("Completed questions appear here.")
    else:
        rows = []
        for i, r in enumerate(results, 1):
            a, c = r.get("audit") or {}, r["citations"]
            rows.append({"#": i, "Time": r["time"], "Question": r["question"], "Abstained": r["abstained"],
                         "Faithfulness %": a.get("faithfulness"),
                         "Citations OK": f"{c['valid']}/{c['valid'] + c['invalid']}",
                         "Ungrounded numbers": ", ".join(r["numbers"]), "Latency s": r["latency"],
                         "Passages": len(r["sources"])})
        df = pd.DataFrame(rows)
        scores = df["Faithfulness %"].dropna()
        x, y, z, w = st.columns(4)
        x.metric("Questions", len(df))
        y.metric("Audited", len(scores))
        z.metric("Avg faithfulness", f"{scores.mean():.0f}%" if len(scores) else "N/A")
        w.metric("Avg latency", f"{df['Latency s'].mean():.1f}s")
        left, right = st.columns(2)
        if len(scores):
            left.plotly_chart(px.line(df.dropna(subset=["Faithfulness %"]), x="#", y="Faithfulness %", markers=True,
                                      range_y=[0, 100], title="Faithfulness per question"), width="stretch")
            tot = {"Supported": 0, "Contradicted": 0, "Unverifiable": 0}
            for r in results:
                if r.get("audit"):
                    for k, v in zip(tot, ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")):
                        tot[k] += r["audit"]["counts"][v]
            right.plotly_chart(px.bar(x=list(tot), y=list(tot.values()), labels={"x": "Verdict", "y": "Claims"},
                                      title="Claim verdicts"), width="stretch")
        st.dataframe(df, width="stretch", hide_index=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ CSV", df.to_csv(index=False).encode(), "hallucination_hunter_history.csv", "text/csv", width="stretch")
        d2.download_button("⬇️ JSON", json.dumps(results, ensure_ascii=False, indent=2, default=str),
                           "hallucination_hunter_report.json", "application/json", width="stretch")
        d3.download_button("⬇️ Markdown report", E.report_markdown(results), "hallucination_hunter_report.md",
                           "text/markdown", width="stretch")

st.caption("Evidence-first answers · The auditor is also an AI model, so verify important claims on the cited PDF pages.")
