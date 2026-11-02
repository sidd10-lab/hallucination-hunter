"""Tests for highlighting, trust score, self-repair, OCR and the Supabase client (all offline)."""
import json
import sys
from pathlib import Path

import numpy as np
try:
    import pymupdf as fitz
except ImportError:
    import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import store as S  # noqa: E402
from test_engine import FakeLLM, build  # noqa: E402


def test_highlight_html_colors_and_escapes():
    answer = "The Pi costs 60 dollars [demo.pdf, p. 1]. It has <b>8</b> cores [demo.pdf, p. 1]."
    claims = [
        {"claim": "costs 60", "span": "The Pi costs 60 dollars [demo.pdf, p. 1].", "verdict": "SUPPORTED",
         "reason": "ok", "quote": "It costs 60 dollars", "quote_verified": True},
        {"claim": "8 cores", "span": "It has <b>8</b> cores", "verdict": "CONTRADICTED",
         "reason": "evidence says 4", "quote": "", "quote_verified": False}]
    h = E.highlight_html(answer, claims)
    assert 'hh-supported' in h and 'hh-contradicted' in h
    assert "<b>8</b>" not in h and "&lt;b&gt;8&lt;/b&gt;" in h  # LLM text is escaped, never trusted as HTML
    assert E.highlight_html("no match here", [{"claim": "zzz qqq", "span": "", "verdict": "SUPPORTED",
                                               "reason": "", "quote": "", "quote_verified": False}]) is None


def test_highlight_fuzzy_sentence_match():
    answer = "Chlorophyll absorbs red and blue light. Plants release oxygen."
    claims = [{"claim": "x", "span": "chlorophyll absorbs mostly red and blue light", "verdict": "UNVERIFIABLE",
               "reason": "r", "quote": "", "quote_verified": False}]
    assert "hh-unverifiable" in E.highlight_html(answer, claims)


def test_trust_score():
    ok = {"invalid": 0, "valid": 2}
    aud = {"faithfulness": 90}
    assert E.trust_score(False, ok, [], aud) == 90
    assert E.trust_score(False, {"invalid": 1, "valid": 1}, ["8"], aud) == 65
    assert E.trust_score(False, {"invalid": 0, "valid": 0}, [], aud) == 70
    assert E.trust_score(True, ok, [], aud) is None and E.trust_score(False, ok, [], None) is None


class RepairLLM(FakeLLM):
    def generate(self, prompt, json_mode=False, temperature=0.0):
        if json_mode:
            self.calls = getattr(self, "calls", 0) + 1
            if "Rewrite" in prompt:
                return ""
            bad = "8 cores" in prompt.split("ANSWER:")[1].split("EVIDENCE:")[0]
            claims = ([{"claim": "costs 60", "span": "costs 60 dollars", "verdict": "SUPPORTED", "source": 1,
                        "quote": "costs 60 dollars", "reason": "ok"}]
                      + ([{"claim": "8 cores", "span": "8 cores", "verdict": "CONTRADICTED", "source": 1,
                           "quote": "", "reason": "wrong"}] if bad else []))
            return json.dumps({"claims": claims, "summary": "s"})
        if "Rewrite the ANSWER" in prompt:
            return "The Pi 5 costs 60 dollars [demo.pdf, p. 1]."
        return "The Pi 5 costs 60 dollars [demo.pdf, p. 1]. It has 8 cores [demo.pdf, p. 1]."


def test_self_repair_improves_trust():
    llm, _, corpus = build(RepairLLM())
    r = E.answer_once(llm, corpus, "How much is the Raspberry Pi 5?", k=3, repair_on=True)
    assert r["audit"]["counts"]["CONTRADICTED"] == 1 and r["trust"] is not None
    rep = r["repaired"]
    assert rep and "8 cores" not in rep["answer"] and rep["audit"]["faithfulness"] == 100
    assert rep["trust"] > r["trust"]
    json.dumps(r, default=str)
    assert "Self-repaired" in E.report_markdown([r])


def test_no_repair_when_all_supported():
    llm, _, corpus = build(RepairLLM())
    llm.reply = "x"
    class Clean(RepairLLM):
        def generate(self, prompt, json_mode=False, temperature=0.0):
            if not json_mode:
                return "The Pi 5 costs 60 dollars [demo.pdf, p. 1]."
            return super().generate(prompt, json_mode, temperature)
    llm, _, corpus = build(Clean())
    r = E.answer_once(llm, corpus, "How much is the Raspberry Pi 5?", k=3, repair_on=True)
    assert r["repaired"] is None and r["audit"]["faithfulness"] == 100


def scanned_pdf() -> bytes:
    src = fitz.open()
    p = src.new_page()
    p.insert_text((50, 100), "Marathi exam notes page", fontsize=20)
    png = p.get_pixmap(matrix=fitz.Matrix(1, 1)).tobytes("png")
    pdf = fitz.open()
    pdf.new_page().insert_image(fitz.Rect(0, 0, 400, 400), stream=png)        # page 1: image only
    pdf.new_page().insert_textbox(fitz.Rect(50, 50, 500, 700),
                                  "This second page has real selectable text about the Eiffel Tower height.", fontsize=12)
    data = pdf.tobytes()
    return data


class OcrLLM(FakeLLM):
    def ocr(self, images, page_numbers):
        assert all(i[:2] == b"\xff\xd8" for i in images)  # JPEG magic bytes
        return {n: f"Scanned text recovered for page {n} about lighthouse keepers and tides" for n in page_numbers}


def test_scanned_detection_and_ocr_in_build_doc():
    data = scanned_pdf()
    pages = E.extract_pages(data)
    assert E.scanned_pages(data, pages) == [1]
    llm = OcrLLM()
    doc = E.build_doc(llm, "scan.pdf", "d", data, 60, 10, ocr=True)
    assert doc["ocr_pages"] == 1 and {c["page"] for c in doc["chunks"]} == {1, 2}
    no_ocr = E.build_doc(llm, "scan.pdf", "d", data, 60, 10, ocr=False)
    assert no_ocr["ocr_pages"] == 0 and {c["page"] for c in no_ocr["chunks"]} == {2}


def test_ocr_parser_splits_pages():
    class Raw(E.GeminiLLM):
        def __init__(self): self.active = "x"; self.models = ["x"]
        def generate(self, prompt, json_mode=False, temperature=0.0):
            return "===PAGE 3===\nhello world one\n===PAGE 7===\nsecond page text"
    out = Raw().ocr([b"a", b"b"], [3, 7])
    assert out == {3: "hello world one", 7: "second page text"}


def test_bundle_roundtrip():
    _, doc, _ = build()
    chunks, emb = S.unpack_bundle(S.pack_bundle(doc["chunks"], doc["emb"]))
    assert chunks == doc["chunks"] and emb.shape == doc["emb"].shape
    assert float(np.abs(emb - doc["emb"]).max()) < 5e-3  # float16 storage is accurate enough
    assert S.owner_id("secret-code") != S.owner_id("secret-code2") and len(S.owner_id("x")) == 32


class Resp:
    def __init__(self, code=200, content=b"", rows=None):
        self.status_code, self.content, self._rows, self.text = code, content, rows, ""
    def json(self): return self._rows


class FakeSession:
    def __init__(self): self.objs, self.rows = {}, {}
    def post(self, url, headers=None, data=None, params=None, timeout=None):
        if "/storage/v1/object/" in url:
            self.objs[url.split("/hh-files/")[1]] = data
            return Resp(200)
        row = json.loads(data)
        self.rows[(row["owner"], row["key"])] = row
        return Resp(201)
    def get(self, url, headers=None, params=None, timeout=None):
        if "/storage/v1/object/" in url:
            return Resp(200, self.objs[url.split("/hh-files/")[1]])
        rows = [r for r in self.rows.values()
                if all(params[k] == f"eq.{r[k]}" for k in ("owner", "embed_model", "emb_dim", "key") if k in params)]
        return Resp(200, rows=rows)
    def delete(self, url, headers=None, data=None, params=None, timeout=None):
        if "/storage/" in url:
            for p in json.loads(data)["prefixes"]:
                self.objs.pop(p, None)
        else:
            self.rows.pop((params["owner"].split(".", 1)[1], params["key"].split(".", 1)[1]), None)
        return Resp(200)


def test_cloud_save_list_load_delete_and_isolation():
    _, doc, _ = build()
    cloud = S.Cloud("https://x.supabase.co", "k", session=FakeSession())
    mine, other = S.owner_id("my-secret-1"), S.owner_id("someone-else")
    key = S.make_key(doc["digest"], 60, 10, "emb-model", 768, False)
    assert not cloud.exists(mine, key)
    cloud.save(mine, doc, key, 60, 10, "emb-model", 768)
    assert cloud.exists(mine, key) and not cloud.exists(other, key)
    rows = cloud.list_docs(mine, "emb-model", 768)
    assert len(rows) == 1 and cloud.list_docs(other, "emb-model", 768) == [] and cloud.list_docs(mine, "other-model", 768) == []
    loaded = cloud.load(mine, rows[0])
    assert loaded["data"] == doc["data"] and loaded["chunks"] == doc["chunks"] and loaded["name"] == "demo.pdf"
    cloud.delete(mine, key)
    assert cloud.list_docs(mine, "emb-model", 768) == []
