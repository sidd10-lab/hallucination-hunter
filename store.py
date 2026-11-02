"""Optional cloud library on Supabase (free tier). No Streamlit imports, so it is testable.

Layout
- table  public.kb_docs : one metadata row per saved PDF (see supabase_schema.sql)
- bucket hh-files       : <owner>/<key>.pdf  (the PDF)  and  <owner>/<key>.npz  (chunks + embeddings)
`owner` is a hash of the user's workspace code, so each code sees only its own library.
Use the service_role key and keep it ONLY in Streamlit Secrets (server side), never in the repo.
"""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import requests

BUCKET = "hh-files"
COLUMNS = ("key,name,digest,pages,words,text_pages,ocr_pages,chunk_count,chunk_size,overlap,"
           "embed_model,emb_dim,size_bytes,created_at")


def owner_id(code: str) -> str:
    return hashlib.sha256(("hh-v1:" + code.strip()).encode()).hexdigest()[:32]


def make_key(digest: str, size: int, overlap: int, embed_model: str, dim: int, ocr: bool) -> str:
    tag = hashlib.sha1(f"{embed_model}|{dim}".encode()).hexdigest()[:6]
    return f"{digest[:20]}_{size}_{overlap}_{int(ocr)}_{tag}"


def pack_bundle(chunks: list[dict], emb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    payload = json.dumps(chunks, ensure_ascii=False).encode("utf-8")
    np.savez_compressed(buf, emb=emb.astype(np.float16), chunks=np.frombuffer(payload, dtype=np.uint8))
    return buf.getvalue()


def unpack_bundle(raw: bytes) -> tuple[list[dict], np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:  # no pickle: safe to load from remote storage
        chunks = json.loads(z["chunks"].tobytes().decode("utf-8"))
        emb = z["emb"].astype(np.float32)
    return chunks, emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)


class Cloud:
    def __init__(self, url: str, key: str, session: requests.Session | None = None, timeout: int = 60):
        self.url = url.rstrip("/")
        self.http = session or requests.Session()
        self.timeout = timeout
        self.h = {"apikey": key, "Authorization": f"Bearer {key}"}

    @staticmethod
    def _ok(r: requests.Response) -> requests.Response:
        if r.status_code >= 300:
            raise RuntimeError(f"Supabase error {r.status_code}: {r.text[:200]}")
        return r

    def _rest(self) -> str:
        return f"{self.url}/rest/v1/kb_docs"

    def _obj(self, path: str) -> str:
        return f"{self.url}/storage/v1/object/{BUCKET}/{path}"

    def ping(self) -> None:
        self._ok(self.http.get(self._rest(), headers=self.h, params={"select": "key", "limit": "1"}, timeout=self.timeout))

    def list_docs(self, owner: str, embed_model: str, dim: int) -> list[dict]:
        r = self._ok(self.http.get(self._rest(), headers=self.h, timeout=self.timeout, params={
            "select": COLUMNS, "owner": f"eq.{owner}", "embed_model": f"eq.{embed_model}",
            "emb_dim": f"eq.{dim}", "order": "created_at.desc"}))
        return r.json()

    def exists(self, owner: str, key: str) -> bool:
        r = self._ok(self.http.get(self._rest(), headers=self.h, timeout=self.timeout,
                                   params={"select": "key", "owner": f"eq.{owner}", "key": f"eq.{key}"}))
        return bool(r.json())

    def _upload(self, path: str, data: bytes, ctype: str) -> None:
        h = {**self.h, "Content-Type": ctype, "x-upsert": "true"}
        self._ok(self.http.post(self._obj(path), headers=h, data=data, timeout=self.timeout))

    def save(self, owner: str, doc: dict, key: str, size: int, overlap: int, embed_model: str, dim: int) -> None:
        self._upload(f"{owner}/{key}.pdf", doc["data"], "application/pdf")
        self._upload(f"{owner}/{key}.npz", pack_bundle(doc["chunks"], doc["emb"]), "application/octet-stream")
        row = {"owner": owner, "key": key, "name": doc["name"], "digest": doc["digest"], "pages": doc["pages"],
               "words": doc["words"], "text_pages": doc["text_pages"], "ocr_pages": doc.get("ocr_pages", 0),
               "chunk_count": len(doc["chunks"]), "chunk_size": size, "overlap": overlap,
               "embed_model": embed_model, "emb_dim": dim, "size_bytes": len(doc["data"])}
        h = {**self.h, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
        self._ok(self.http.post(self._rest(), headers=h, params={"on_conflict": "owner,key"},
                                data=json.dumps(row), timeout=self.timeout))

    def load(self, owner: str, row: dict) -> dict:
        key = row["key"]
        pdf = self._ok(self.http.get(self._obj(f"{owner}/{key}.pdf"), headers=self.h, timeout=self.timeout)).content
        bundle = self._ok(self.http.get(self._obj(f"{owner}/{key}.npz"), headers=self.h, timeout=self.timeout)).content
        chunks, emb = unpack_bundle(bundle)
        return {"name": row["name"], "digest": row["digest"], "data": pdf, "chunks": chunks, "emb": emb,
                "pages": row["pages"], "words": row["words"], "text_pages": row["text_pages"],
                "ocr_pages": row.get("ocr_pages") or 0, "cloud_key": key}

    def delete(self, owner: str, key: str) -> None:
        self._ok(self.http.delete(f"{self.url}/storage/v1/object/{BUCKET}", headers={**self.h, "Content-Type": "application/json"},
                                  data=json.dumps({"prefixes": [f"{owner}/{key}.pdf", f"{owner}/{key}.npz"]}), timeout=self.timeout))
        self._ok(self.http.delete(self._rest(), headers=self.h, timeout=self.timeout,
                                  params={"owner": f"eq.{owner}", "key": f"eq.{key}"}))
