"""Run with:  python -m pytest -q   (no API key or network needed)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
try:
    import pymupdf as fitz
except ImportError:
    import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine as E  # noqa: E402


class FakeLLM(E.GeminiLLM):
    """Bag-of-words 'embeddings' and canned generations, so the whole pipeline runs offline."""

    def __init__(self):
        self.active = "fake"
        self.models = ["fake"]
        self.embed_model = "fake"
        self.reply = ""
        self.audit_json = {"claims": [], "summary": ""}

    def embed(self, texts, task, progress=None):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in E.tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-10)

    def generate(self, prompt, json_mode=False, temperature=0.0):
        return json.dumps(self.audit_json) if json_mode else self.reply


def make_pdf(pages: list[str]) -> bytes:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return data


PAGES = [
    "The Raspberry Pi 5 was released in October 2023. It costs 60 dollars for the 4 GB model. "
    "The board uses a quad-core Arm Cortex-A76 processor running at 2.4 GHz.",
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly red and blue light. "
    "Plants release oxygen as a by-product of the process.",
    "The Eiffel Tower is 330 metres tall and was completed in 1889 for the World's Fair in Paris.",
]


def build(llm=None):
    llm = llm or FakeLLM()
    data = make_pdf(PAGES)
    doc = E.build_doc(llm, "demo.pdf", "d1", data, 60, 10)
    return llm, doc, E.Corpus([doc])


def test_tokenize_keeps_devanagari():
    toks = E.tokenize("महाराष्ट्र लोकसेवा आयोग परीक्षा")
    assert "महाराष्ट्र" in toks and len(toks) == 4


def test_chunks_have_pages_and_size_limit():
    _, doc, _ = build()
    assert doc["pages"] == 3 and {c["page"] for c in doc["chunks"]} == {1, 2, 3}
    assert all(len(c["text"].split()) <= 60 for c in doc["chunks"])


def test_pack_sentences_overlap_and_long_sentence():
    chunks = E.pack_sentences(["a b c d.", "e f g h.", "i j k l."], size=8, overlap=4)
    assert len(chunks) == 2 and "e f g h." in chunks[0] and "e f g h." in chunks[1]
    assert all(len(c.split()) <= 10 for c in E.pack_sentences([" ".join(["w"] * 25)], 10, 2))


def test_hybrid_retrieval_finds_right_page():
    llm, _, corpus = build()
    for q, page in [("How much does the Raspberry Pi 5 cost?", 1), ("What does chlorophyll absorb?", 2),
                    ("How tall is the Eiffel Tower?", 3)]:
        sources, _ = corpus.retrieve(llm, q, k=2)
        assert sources[0]["page"] == page, q
    sources, _ = corpus.retrieve(llm, "Eiffel", k=3, mmr=False)
    assert len({(s["file"], s["page"], s["text"]) for s in sources}) == len(sources)


def test_bm25_prefers_rare_terms():
    bm = E.BM25(["alpha beta gamma", "alpha alpha beta", "delta zeta gamma"])
    s = bm.scores(E.tokenize("delta"))
    assert s.argmax() == 2 and s[0] == 0


def test_parse_and_check_citations():
    answer = "Costs 60 dollars [demo.pdf, p. 1]. Tall [demo.pdf, p. 3; other.pdf, p. 2]. Range [demo.pdf, pp. 1-2]."
    assert ("demo.pdf", 1) in E.parse_citations(answer) and ("other.pdf", 2) in E.parse_citations(answer)
    _, _, corpus = build()
    sources = [{"file": "demo.pdf", "page": 1, "text": "x"}, {"file": "demo.pdf", "page": 3, "text": "y"}]
    res = E.check_citations(answer, sources)
    bad = {(i["file"], i["page"]) for i in res["items"] if not i["valid"]}
    assert bad == {("other.pdf", 2), ("demo.pdf", 2)} and res["valid"] == 2


def test_ungrounded_numbers():
    sources = [{"file": "f", "page": 1, "text": "It costs 1,200 dollars and weighs 3.5 kg."}]
    answer = "It costs 1200 dollars [f, p. 1] and weighs 4.5 kg.\n1. First point"
    assert E.ungrounded_numbers(answer, sources, "how much?") == ["4.5"]


def test_quote_coverage():
    text = "Chlorophyll absorbs mostly red and blue light."
    assert E.quote_coverage("chlorophyll absorbs mostly RED and blue light", text) == 1.0
    assert E.quote_coverage("chlorophyll absorbs green light only", text) < 0.85


def test_audit_downgrades_fake_quotes():
    llm, _, corpus = build()
    sources, _ = corpus.retrieve(llm, "chlorophyll", k=2)
    llm.audit_json = {"claims": [
        {"claim": "Chlorophyll absorbs red and blue light", "verdict": "SUPPORTED", "source": 1,
         "quote": "Chlorophyll absorbs mostly red and blue light", "reason": "stated"},
        {"claim": "Chlorophyll is found in mitochondria", "verdict": "SUPPORTED", "source": 1,
         "quote": "Chlorophyll lives in mitochondria of every cell", "reason": "judge lied"},
        {"claim": "Plants emit helium", "verdict": "CONTRADICTED", "source": None, "quote": "", "reason": "no"}],
        "summary": "mixed"}
    a = E.audit(llm, "q", "ans", sources)
    assert a["counts"] == {"SUPPORTED": 1, "CONTRADICTED": 1, "UNVERIFIABLE": 1}
    assert a["downgraded"] == 1 and a["faithfulness"] == 33
    assert a["claims"][0]["quote_verified"] and a["claims"][0]["page"] == 2


def test_answer_once_end_to_end_and_abstain():
    llm, _, corpus = build()
    llm.reply = "The Pi 5 costs 60 dollars [demo.pdf, p. 1]. It has 8 cores [demo.pdf, p. 2]."
    llm.audit_json = {"claims": [], "summary": "ok"}
    r = E.answer_once(llm, corpus, "How much is the Raspberry Pi 5?", k=3)
    assert not r["abstained"] and r["numbers"] == ["8"] and r["citations"]["invalid"] in (0, 1)
    json.dumps(r, default=str)
    llm.reply = E.ABSTAIN
    r = E.answer_once(llm, corpus, "Who won the 2030 World Cup?", k=3)
    assert r["abstained"] and r["audit"] is None
    r = E.answer_once(llm, corpus, "Anything", k=3, min_rel=0.99)
    assert r["abstained"]


def test_classify():
    abst = {"abstained": True}
    assert E.classify("trap", abst) == "Correctly abstained" and E.classify("answerable", abst) == "Wrongly abstained"
    bad = {"abstained": False, "audit": {"counts": {"CONTRADICTED": 0, "UNVERIFIABLE": 1}}}
    assert E.classify("trap", bad) == "Unsupported claims"


def test_render_page_and_summaries_helpers():
    _, doc, corpus = build()
    png = E.render_page(doc["data"], 2, ["Chlorophyll absorbs mostly red and blue light"])
    assert png[:4] == b"\x89PNG"
    assert len(corpus.sample(5)) <= 5
    llm = FakeLLM()
    llm.reply = "# Overview"
    text, used = E.summarize_doc(llm, doc)
    assert text and used >= 1
