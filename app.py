from __future__ import annotations

import hashlib
import html
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
import store as S

load_dotenv()
MAX_FILE_MB, MAX_PAGES, MAX_OCR_PAGES = 40, 700, 40
VIEWS = ["💬 Chat", "📝 Summaries", "🧪 Stress test", "📊 Analytics", "☁️ Library"]

st.set_page_config(page_title="Hallucination Hunter", page_icon="🔎", layout="wide")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        value = ""
    return str(value or os.getenv(name, default)).strip()


API_KEY = secret("GEMINI_API_KEY")
GEN_MODEL = secret("GEN_MODEL", "gemini-3.6-flash")
EMBED_MODEL = secret("EMBED_MODEL", "gemini-embedding-001")
SUPA_URL, SUPA_KEY = secret("SUPABASE_URL"), secret("SUPABASE_KEY")

st.markdown("""
<style>
.hero{padding:2rem 2rem 1.6rem;border:1px solid #2d4166;border-radius:22px;margin-bottom:1.2rem;
background:radial-gradient(circle at 88% 12%,#3b2a70,#1a2848 48%,#14213a)}
.hero h1{margin:.2rem 0 .3rem;padding:0;font-size:2.6rem}
.eyebrow{color:#9db8ef;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700}
.topbar{display:flex;align-items:baseline;gap:.8rem;margin:-.6rem 0 .6rem}
.topbar b{font-size:1.5rem}
.muted{color:#a9b8d4}
.pills{display:flex;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 .9rem}
.pill{background:#121e34;border:1px solid #293d60;border-radius:12px;padding:.4rem .85rem;font-size:.78rem;color:#a9b8d4}
.pill b{display:block;font-size:1.1rem;color:#eef3ff}
.card{background:#111a2e;border:1px solid #27395c;border-radius:16px;padding:1rem 1.1rem;height:100%}
.card h4{margin:.1rem 0 .35rem}
.chip{display:inline-block;padding:.16rem .62rem;border-radius:999px;font-size:.78rem;border:1px solid;margin:.1rem .3rem .1rem 0}
.chip-ok{color:#86efac;border-color:#1f6f3a;background:#0f2a1a}
.chip-warn{color:#fcd34d;border-color:#7a5b10;background:#2b2108}
.chip-bad{color:#fca5a5;border-color:#7a2323;background:#2d1010}
.chip-info{color:#a5c4ff;border-color:#2c4a86;background:#0f1c36}
.trust{display:inline-block;padding:.22rem .8rem;border-radius:999px;font-weight:800;font-size:.9rem;margin:.1rem .4rem .1rem 0}
.trust-good{background:#14532d;color:#bbf7d0}.trust-mid{background:#713f12;color:#fde68a}
.trust-bad{background:#7f1d1d;color:#fecaca}.trust-na{background:#1e293b;color:#94a3b8}
.hh-answer{line-height:1.75}
mark.hh{padding:.04rem .22rem;border-radius:6px;color:inherit;cursor:help}
mark.hh-supported{background:rgba(74,222,128,.22)}
mark.hh-contradicted{background:rgba(248,113,113,.32)}
mark.hh-unverifiable{background:rgba(250,204,21,.26)}
</style>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("Gemini API key is missing. Add GEMINI_API_KEY to your local .env or Streamlit Cloud Secrets.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llm(key: str, gen_model: str, embed_model: str) -> E.GeminiLLM:
    return E.GeminiLLM(key, gen_model, embed_model)


@st.cache_resource(show_spinner=False)
def get_cloud(url: str, key: str) -> S.Cloud:
    return S.Cloud(url, key)


@st.cache_data(ttl=60, show_spinner=False)
def list_library(owner: str, embed_model: str, dim: int, _cloud) -> list[dict]:
    return _cloud.list_docs(owner, embed_model, dim)


llm = get_llm(API_KEY, GEN_MODEL, EMBED_MODEL)
cloud = get_cloud(SUPA_URL, SUPA_KEY) if (SUPA_URL and SUPA_KEY) else None


class DocStore:
    """Shared in-memory LRU of indexed PDFs, keyed by content (nothing is written to disk)."""

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
        return "Gemini quota / rate limit reached. Wait a minute, or turn off Audit / Self-repair in Settings."
    if "503" in m or "UNAVAILABLE" in m or "OVERLOADED" in m:
        return "Gemini is temporarily busy. Wait 10–20 seconds and retry."
    if "API_KEY" in m or "PERMISSION_DENIED" in m or "401" in m or "403" in m:
        return "The API key looks invalid or lacks access. Check GEMINI_API_KEY."
    if "404" in m or "NOT_FOUND" in m:
        return "Model not found. Check GEN_MODEL / EMBED_MODEL in .env or Secrets."
    return ""


# --------------------------------------------------------------------------- small UI helpers
def chip(text: str, kind: str = "info") -> str:
    return f'<span class="chip chip-{kind}">{html.escape(text)}</span>'


def trust_badge(score: int | None, label: str = "Trust") -> str:
    if score is None:
        return f'<span class="trust trust-na">{label} –</span>'
    kind = "good" if score >= 85 else "mid" if score >= 60 else "bad"
    return f'<span class="trust trust-{kind}">🛡 {label} {score}</span>'


for k, v in {"messages": [], "pending": None, "summaries": {}, "stress": None, "suggestions": None,
             "signature": None, "view": VIEWS[0], "saved_keys": set()}.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------------------------------- sidebar
owner, lib_rows, autosave = None, [], False
with st.sidebar:
    st.markdown("## 🔎 Hallucination Hunter")
    st.markdown("#### 📚 Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    ocr_on = st.toggle("Read scanned pages (OCR)", False,
                       help=f"Uses Gemini vision on pages with no selectable text (up to {MAX_OCR_PAGES} pages per PDF).")

    if cloud:
        st.markdown("#### ☁️ My library")
        code = st.text_input("Workspace code", type="password", placeholder="your private code",
                             help="Your saved PDFs are tied to this code. Use a long, secret one (6+ characters).")
        if code and len(code.strip()) >= 6:
            owner = S.owner_id(code)
        elif code:
            st.caption("Use at least 6 characters.")
        if owner:
            try:
                lib_rows = list_library(owner, EMBED_MODEL, llm.dim, cloud)
            except Exception as e:  # noqa: BLE001
                st.error(f"Library unavailable: {str(e)[:150]}")
            valid = {r["key"] for r in lib_rows}
            st.session_state["lib_pick"] = [k for k in st.session_state.get("lib_pick", []) if k in valid]
            fmt = {r["key"]: f"{r['name']} · {r.get('pages') or 0}p" for r in lib_rows}
            st.multiselect("Open saved PDFs", list(fmt), key="lib_pick", format_func=lambda k: fmt[k],
                           placeholder="Choose from your library")
            autosave = st.toggle("Save new uploads to my library", True)
    scope_box = st.container()

    with st.expander("⚙️ Settings"):
        top_k = st.slider("Evidence passages", 3, 12, 6)
        chunk_size = st.slider("Passage size (words)", 120, 400, 220, step=20)
        overlap = st.slider("Passage overlap (words)", 0, 100, 40, step=10)
        audit_on = st.toggle("Audit every answer", True,
                             help="Checks each claim and verifies its quote against the PDF text.")
        repair_on = st.toggle("Self-repair bad answers", True,
                              help="If the audit finds unsupported claims, rewrites the answer and re-audits it. One or two extra calls.")
        rewrite_on = st.toggle("Understand follow-ups", True)
        mmr_on = st.toggle("Diverse evidence (MMR)", True)
        min_rel = st.slider("Abstain below relevance", 0.0, 0.8, 0.0, step=0.05,
                            help="If even the best passage is less similar than this, skip the model and abstain. 0 = off.")
        st.caption(f"Answer model: {llm.active} · Embeddings: {EMBED_MODEL}")
    if st.button("Clear chat and results", width="stretch"):
        st.session_state.update(messages=[], suggestions=None, stress=None, summaries={}, pending=None)
        st.rerun()

# --------------------------------------------------------------------------- landing page
if not uploads and not st.session_state.get("lib_pick"):
    st.markdown("""<div class="hero"><div class="eyebrow">Document-grounded AI reliability</div>
<h1>🔎 Hallucination Hunter</h1>
<div class="muted" style="font-size:1.05rem">Ask your PDFs anything. Every answer is fact-checked against the source,
highlighted sentence by sentence, and automatically repaired if it makes things up.</div></div>""", unsafe_allow_html=True)
    cards = [("🎯", "Claim-level audit", "Each claim must be backed by a quote that really exists in your PDF."),
             ("🖍️", "Live highlighting", "Green, red and yellow sentences show exactly what to trust."),
             ("🛠️", "Self-repair", "Unsupported claims are removed and the new answer is re-audited."),
             ("📷", "Scanned PDFs", "Turn on OCR to read photographed notes, including Marathi and Hindi.")]
    for col, (icon, title, text) in zip(st.columns(4), cards):
        col.markdown(f'<div class="card"><h4>{icon} {title}</h4><div class="muted">{text}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("##### How it works")
    a, b, c = st.columns(3)
    a.markdown("**1. Upload** one or more PDFs in the left panel.")
    b.markdown("**2. Ask** a question in the chat box.")
    c.markdown("**3. Check** the trust score, highlights and cited pages.")
    if cloud:
        st.caption("Have saved PDFs? Enter your workspace code in the sidebar and open them from My library.")
    st.stop()

# --------------------------------------------------------------------------- load documents
store = doc_store()
loaded, names_seen, total_pages = [], set(), 0


def unique_name(name: str) -> str:
    base, n = name, 2
    while name in names_seen:
        name = base.replace(".pdf", "") + f" ({n}).pdf"
        n += 1
    names_seen.add(name)
    return name


for up in uploads or []:
    name = unique_name(up.name)
    data = up.getvalue()
    if len(data) > MAX_FILE_MB * 2**20:
        st.error(f"{name} exceeds {MAX_FILE_MB} MB.")
        continue
    digest = hashlib.sha256(data).hexdigest()
    key = (name, digest, chunk_size, overlap, EMBED_MODEL, ocr_on)
    doc = store.get(key)
    if doc is None:
        bar = st.progress(0.0, text=f"Indexing {name}…")
        try:
            pages = E.page_count(data)
            if total_pages + pages > MAX_PAGES:
                st.error(f"Combined page limit is {MAX_PAGES}; {name} would exceed it.")
                continue
            doc = E.build_doc(llm, name, digest, data, chunk_size, overlap,
                              lambda f, label, n=name: bar.progress(min(1.0, max(0.0, f)), text=f"{label} · {n} · {int(f * 100)}%"),
                              ocr=ocr_on, max_ocr=MAX_OCR_PAGES)
            store.put(key, doc)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not index {name}: {e}")
            continue
        finally:
            bar.empty()
    total_pages += doc["pages"]
    loaded.append(doc)
    if cloud and owner and autosave:  # persist to the free Supabase library once per session
        skey = S.make_key(digest, chunk_size, overlap, EMBED_MODEL, llm.dim, ocr_on)
        if skey not in st.session_state.saved_keys:
            try:
                if not cloud.exists(owner, skey):
                    with st.spinner(f"Saving {name} to your library…"):
                        cloud.save(owner, doc, skey, chunk_size, overlap, EMBED_MODEL, llm.dim)
                    list_library.clear()
                st.session_state.saved_keys.add(skey)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not save {name} to your library: {str(e)[:150]}")

if cloud and owner:
    rows_by_key = {r["key"]: r for r in lib_rows}
    for lk in st.session_state.get("lib_pick", []):
        row = rows_by_key.get(lk)
        if not row or any(d["digest"] == row["digest"] for d in loaded):
            continue
        skey = ("lib", owner, lk)
        doc = store.get(skey)
        if doc is None:
            try:
                with st.spinner(f"Opening {row['name']}…"):
                    doc = {**cloud.load(owner, row)}
                store.put(skey, doc)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not open {row['name']}: {str(e)[:150]}")
                continue
        doc = {**doc, "name": unique_name(doc["name"])}
        total_pages += doc["pages"]
        loaded.append(doc)

if not loaded:
    st.stop()
for d in loaded:
    if d["text_pages"] < 0.6 * d["pages"] and not d.get("ocr_pages"):
        st.warning(f"{d['name']}: only {d['text_pages']} of {d['pages']} pages have selectable text. "
                   "Turn on **Read scanned pages (OCR)** in the sidebar and upload it again.")

with scope_box:
    chosen = st.multiselect("Search in", [d["name"] for d in loaded], default=[d["name"] for d in loaded])
docs = [d for d in loaded if d["name"] in chosen]
if not docs:
    st.warning("Select at least one document to search.")
    st.stop()
by_name = {d["name"]: d for d in docs}

sig = tuple((d["name"], d["digest"]) for d in docs) + (chunk_size, overlap, ocr_on)
if st.session_state.signature != sig:  # documents or chunking changed, so old answers no longer apply
    st.session_state.signature = sig
    st.session_state.update(messages=[], suggestions=None, stress=None, summaries={}, pending=None)
corpus = get_corpus(json.dumps(sig), docs)

ocr_total = sum(d.get("ocr_pages", 0) for d in docs)
st.markdown(
    '<div class="topbar"><b>🔎 Hallucination Hunter</b><span class="muted">grounded answers you can verify</span></div>'
    '<div class="pills">'
    f'<div class="pill">PDFs<b>{len(docs)}</b></div><div class="pill">Pages<b>{sum(d["pages"] for d in docs)}</b></div>'
    f'<div class="pill">Passages<b>{len(corpus.chunks)}</b></div><div class="pill">Words<b>{sum(d["words"] for d in docs):,}</b></div>'
    + (f'<div class="pill">OCR pages<b>{ocr_total}</b></div>' if ocr_total else "") + "</div>", unsafe_allow_html=True)

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


def answer_html(answer: str, audit_res: dict | None, abstained: bool) -> str | None:
    if abstained or not audit_res:
        return None
    return E.highlight_html(answer, audit_res["claims"])


def checks_chips(abstained: bool, cit: dict, numbers: list[str], a: dict | None) -> str:
    chips = []
    if abstained:
        return chip("Abstained: not in the document", "info")
    total_c = cit["valid"] + cit["invalid"]
    if total_c == 0:
        chips.append(chip("No page citations", "bad"))
    elif cit["invalid"]:
        chips.append(chip(f"{cit['invalid']}/{total_c} citations point to unseen pages", "bad"))
    else:
        chips.append(chip(f"{total_c}/{total_c} citations match", "ok"))
    chips.append(chip("Numbers not in evidence: " + ", ".join(numbers), "warn") if numbers else chip("Numbers grounded", "ok"))
    if a:
        c = a["counts"]
        chips += [chip(f"{c['SUPPORTED']} supported", "ok"),
                  chip(f"{c['CONTRADICTED']} contradicted", "bad" if c["CONTRADICTED"] else "info"),
                  chip(f"{c['UNVERIFIABLE']} unverifiable", "warn" if c["UNVERIFIABLE"] else "info")]
    return " ".join(chips)


def show_answer_text(text: str, h: str | None) -> None:
    if h:
        st.markdown(h, unsafe_allow_html=True)
    else:
        st.markdown(text)


def render_result(res: dict, key: str, show_answer: bool = False) -> None:
    a, rep = res.get("audit"), res.get("repaired")
    h = answer_html(res["answer"], a, res["abstained"])
    if show_answer:
        show_answer_text(res["answer"], h)
    st.markdown(trust_badge(res.get("trust")) + " " + checks_chips(res["abstained"], res["citations"], res["numbers"], a),
                unsafe_allow_html=True)
    if h:
        st.caption("Hover a highlighted sentence to see why. 🟩 supported · 🟥 contradicted · 🟨 unverifiable")
    if a and a["downgraded"]:
        st.caption(f"🔎 {a['downgraded']} claim(s) were downgraded because the auditor's quote was not found in the PDF text.")
    if res.get("audit_error"):
        st.warning("Answer succeeded, but the audit failed: " + res["audit_error"])
    bad = [f"{i['file']} p. {i['page']}" for i in res["citations"]["items"] if not i["valid"]]
    if bad and not res["abstained"]:
        st.warning("Citations to pages that were not in the evidence: " + ", ".join(bad))

    if rep:
        with st.container(border=True):
            st.markdown("##### 🛠 Self-repaired answer")
            st.markdown(trust_badge(res.get("trust"), "Before") + "→ " + trust_badge(rep.get("trust"), "After"),
                        unsafe_allow_html=True)
            show_answer_text(rep["answer"], answer_html(rep["answer"], rep.get("audit"), rep["abstained"]))
            if rep["abstained"]:
                st.caption("Nothing in the original answer could be supported by the document.")
            elif rep["numbers"]:
                st.caption("⚠️ Numbers still not found in evidence: " + ", ".join(rep["numbers"]))
    elif res.get("repair_error"):
        st.caption("Self-repair failed: " + res["repair_error"][:120])

    if a and a["claims"]:
        with st.expander("🔬 Claim-by-claim audit", expanded=False):
            st.write(a["summary"])
            df = pd.DataFrame(a["claims"])[["claim", "verdict", "file", "page", "quote", "reason"]]
            st.dataframe(df, width="stretch", hide_index=True)

    marks: dict = {}
    for cl in (a or {}).get("claims", []):
        if cl["quote_verified"]:
            marks.setdefault((cl["file"], cl["page"]), []).append(cl["quote"])
    with st.expander(f"📚 Evidence ({len(res['sources'])} passages) · {res['latency']}s · {res['model']}"):
        st.caption("Similarity is not a truth score. Read the passage or open the PDF page to check.")
        for i, s in enumerate(res["sources"], 1):
            st.markdown(f"**{i}. {s['file']} · page {s['page']}**")
            st.progress(max(0.0, min(1.0, s["score"])), text=f"relevance · semantic {s['semantic']:.2f} · keyword {s['keyword']:.2f}")
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
            slot = st.empty()
            if top < min_rel:
                answer = E.ABSTAIN
                slot.markdown(answer)
            else:
                with slot.container():
                    answer = st.write_stream(llm.stream(E.answer_prompt(q, sources, chat)))
            label = ("Auditing claims" + (" and repairing" if repair_on else "") + "…") if audit_on else "Checking citations…"
            with st.spinner(label):
                res = E.finalize(llm, q, query, answer, sources, top, audit_on, t0, repair_on and audit_on)
            h = answer_html(answer, res["audit"], res["abstained"])
            if h:
                slot.markdown(h, unsafe_allow_html=True)
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
        st.markdown("##### Not sure what to ask?")
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
            if m.get("result"):
                render_result(m["result"], f"m{i}", show_answer=True)
            else:
                st.markdown(m["content"])
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

elif view == VIEWS[3]:
    st.subheader("Analytics & export")
    results = [m["result"] for m in st.session_state.messages if m.get("result")]
    if not results:
        st.info("Completed questions appear here.")
    else:
        rows = []
        for i, r in enumerate(results, 1):
            a, c = r.get("audit") or {}, r["citations"]
            rp = r.get("repaired") or {}
            rows.append({"#": i, "Time": r["time"], "Question": r["question"], "Abstained": r["abstained"],
                         "Trust": r.get("trust"), "Trust after repair": rp.get("trust"),
                         "Faithfulness %": a.get("faithfulness"),
                         "Citations OK": f"{c['valid']}/{c['valid'] + c['invalid']}",
                         "Ungrounded numbers": ", ".join(r["numbers"]), "Latency s": r["latency"],
                         "Passages": len(r["sources"])})
        df = pd.DataFrame(rows)
        trust = df["Trust"].dropna()
        fixed = df.dropna(subset=["Trust", "Trust after repair"])
        x, y, z, w = st.columns(4)
        x.metric("Questions", len(df))
        y.metric("Avg trust", f"{trust.mean():.0f}" if len(trust) else "N/A")
        z.metric("Repaired answers", len(fixed),
                 delta=f"+{(fixed['Trust after repair'] - fixed['Trust']).mean():.0f} avg trust" if len(fixed) else None)
        w.metric("Avg latency", f"{df['Latency s'].mean():.1f}s")
        left, right = st.columns(2)
        if len(trust):
            left.plotly_chart(px.line(df.dropna(subset=["Trust"]), x="#", y="Trust", markers=True,
                                      range_y=[0, 100], title="Trust score per question"), width="stretch")
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

else:
    st.subheader("☁️ My library")
    if not cloud:
        st.info("The cloud library is off. It uses a free Supabase project to keep your indexed PDFs between visits, "
                "so you never re-embed them.")
        st.markdown("1. Create a free project at supabase.com\n"
                    "2. Open **SQL Editor**, paste the contents of `supabase_schema.sql`, click **Run**\n"
                    "3. Copy **Project URL** and the **service_role** key (Project Settings → API)\n"
                    "4. Add `SUPABASE_URL` and `SUPABASE_KEY` to your Streamlit Secrets and reboot")
    elif not owner:
        st.info("Enter your workspace code in the sidebar to see and manage your saved PDFs.")
    elif not lib_rows:
        st.info("Nothing saved yet. Upload a PDF with **Save new uploads to my library** turned on.")
    else:
        st.caption(f"{len(lib_rows)} saved PDF(s). Open them from the sidebar without re-indexing.")
        for r in lib_rows:
            with st.container(border=True):
                c1, c2 = st.columns([6, 1])
                extra = f" · {r['ocr_pages']} OCR pages" if r.get("ocr_pages") else ""
                c1.markdown(f"**{r['name']}**  \n<span class='muted'>{r.get('pages') or 0} pages · {r.get('chunk_count') or 0} passages · "
                            f"{(r.get('words') or 0):,} words{extra} · {(r.get('size_bytes') or 0) / 1e6:.1f} MB · "
                            f"saved {str(r.get('created_at') or '')[:10]}</span>", unsafe_allow_html=True)
                if c2.button("🗑 Delete", key=f"del_{r['key']}"):
                    try:
                        cloud.delete(owner, r["key"])
                        list_library.clear()
                        st.session_state.saved_keys.discard(r["key"])
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Delete failed: {str(e)[:150]}")

st.caption("Evidence-first answers · The auditor is also an AI model, so verify important claims on the cited PDF pages.")
