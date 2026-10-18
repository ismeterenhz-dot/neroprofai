# -*- coding: utf-8 -*-
"""
NeroProf AI v3 - Nöroloji Klinik Karar-Destek Asistanı (7 Katmanlı Mimari v3)

Mimari:
  * Kütüphane     : PDF/DOCX/TXT/MD -> embedding -> SQLite+numpy vektör deposu (kalıcı RAG)
  * Güncel Literatür: PubMed E-utilities (ücretsiz) - YENİ / ESKİ kanıt ayrımıyla
  * Hafıza        : SQLite (vakalar, mesajlar, düşünce defteri, inanç/iddia defteri)
  * Akıl Yürütme  : 7 Katmanlı Klinik Akıl Yürütme Mimarisi v3 (omni_neuro_architecture.md)
  * Kota Optimizasyonu: Otomatik model geçişi, Retry-After okuma, yerel embedding seçeneği

Çalıştırma: streamlit run neroprof_app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="NeroProf AI v3", page_icon="🧠", layout="wide")

# =====================================================================
# 1. SABİTLER VE YAPILANDIRMA
# =====================================================================
BASE = Path(__file__).resolve().parent
LIB_DIR = BASE / "kutuphane"
DATA_DIR = BASE / "neroprof_data"
DB_PATH = DATA_DIR / "neroprof.db"
CONFIG_FILE = BASE / "config_npro.json"
ARCH_FILE = LIB_DIR / "omni_neuro_architecture.md"

for _d in (LIB_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LIB_EXT = (".pdf", ".txt", ".md", ".docx")
IMG_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
DOC_CHAR_CAP = 60000        # Vaka belgelerinden prompta girecek azami karakter
ARCH_CHAR_CAP = 18000       # Mimari belgeden alınacak azami karakter
CLAIM_SIM_MIN = 0.70        # İddia karşılaştırması benzerlik eşiği
YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
FIG_RE = re.compile(r"\b(fig(?:ure|\.)?|şekil|resim|plate)\s*\d", re.I)
FIG_FRAC_MIN = 0.10

EMBED_MODELS = {
    "Gemini embedding (API kotası harcar)": "gemini:gemini-embedding-001",
    "Yerel: multilingual-e5-small (kota yok, hızlı, 512 token) - önerilen": "intfloat/multilingual-e5-small",
    "Yerel: multilingual-MiniLM-L12 (kota yok, en hafif, 128 token)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Yerel: BAAI/bge-m3 (kota yok, yüksek kalite, ağır)": "BAAI/bge-m3",
}

EMBED_MAX_SEQ = {"intfloat/multilingual-e5-small": 512, "BAAI/bge-m3": 512}
EMBED_CHUNKS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (450, 80),
    "intfloat/multilingual-e5-small": (1000, 150),
}

def chunk_params() -> tuple[int, int]:
    name = CFG["embed_model"]
    if name in EMBED_CHUNKS:
        return EMBED_CHUNKS[name]
    return int(CFG["chunk_chars"]), int(CFG["chunk_overlap"])

DEFAULT_CFG = {
    "api_key": "",
    "fast_model": "gemini-3.8-flash",
    "pro_model": "gemini-3.8-flash",
    "embed_model": "gemini:gemini-embedding-001",
    "gemini_embed": "gemini-embedding-001",
    "chunk_chars": 1200,
    "chunk_overlap": 200,
    "ncbi_key": "",
    "ncbi_email": "",
    "mask_pii": True,
}

MODES = {
    "Hızlı":    {"model": "fast", "think": "düşük",  "critic": False, "claims": False, "k_lib": 8},
    "Standart": {"model": "pro",  "think": "orta",   "critic": False, "claims": False, "k_lib": 12},
    "Derin":    {"model": "pro",  "think": "yüksek", "critic": True,  "claims": True,  "k_lib": 16},
}

def load_cfg() -> dict:
    cfg = dict(DEFAULT_CFG)
    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if "api_key" not in raw and raw.get("api_keys"):
                ks = [k for k in raw["api_keys"] if k]
                if ks:
                    raw["api_key"] = ks[0]
            cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CFG})
        except Exception:
            pass
    return cfg

def save_cfg(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except Exception:
        pass

if "cfg" not in st.session_state:
    st.session_state.cfg = load_cfg()
CFG: dict = st.session_state.cfg
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or CFG.get("api_key", "")

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =====================================================================
# 2. VERİTABANI (SQLite)
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, created TEXT, archived INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, role TEXT, content TEXT, sources TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS case_docs(id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, name TEXT, text TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS insights(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, claim TEXT, topic TEXT, source_type TEXT, source_ref TEXT, year INTEGER, evidence TEXT, status TEXT DEFAULT 'aktif', note TEXT);
CREATE TABLE IF NOT EXISTS claim_links(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, new_id INTEGER, old_id INTEGER, relation TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS indexed_files(name TEXT, embed TEXT, sha1 TEXT, chunks INTEGER, empty_pages INTEGER, total_pages INTEGER, year INTEGER, ts TEXT, PRIMARY KEY(name, embed));
CREATE TABLE IF NOT EXISTS usage(ts TEXT, model TEXT, tin INTEGER, tout INTEGER);
CREATE TABLE IF NOT EXISTS vec(coll TEXT, id TEXT, doc TEXT, meta TEXT, src TEXT, emb BLOB, PRIMARY KEY(coll, id));
CREATE INDEX IF NOT EXISTS vec_src ON vec(coll, src);
CREATE TABLE IF NOT EXISTS embedding_cache(
    cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, task TEXT NOT NULL,
    dim INTEGER NOT NULL, emb BLOB NOT NULL, ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS embedding_cache_model ON embedding_cache(model, task);
"""

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def q(sql: str, args: tuple = (), one: bool = False):
    with db() as c:
        rows = c.execute(sql, args).fetchall()
    if one:
        return rows[0] if rows else None
    return rows

def x(sql: str, args: tuple = ()) -> int:
    with db() as c:
        return c.execute(sql, args).lastrowid

def init_db() -> None:
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)

def get_meta(k: str, default: str = "") -> str:
    r = q("SELECT v FROM meta WHERE k=?", (k,), one=True)
    return r["v"] if r else default

def set_meta(k: str, v: str) -> None:
    x("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))

init_db()

# =====================================================================
# 3. METİN ÇIKARMA, PARÇALAMA, MASKELEME
# =====================================================================
def _virtual_pages(text: str, size: int = 3000):
    n = max(1, -(-len(text) // size))
    return [(i + 1, text[i * size:(i + 1) * size]) for i in range(n)]

def _figure_frac(page) -> float:
    try:
        import pypdfium2.raw as pdfium_c
        w, h = page.get_size()
        area = max(float(w) * float(h), 1.0)
        tot = 0.0
        for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=2):
            getb = getattr(obj, "get_bounds", None) or obj.get_pos
            left, bottom, right, top = getb()
            tot += max(0.0, right - left) * max(0.0, top - bottom)
        return min(tot / area, 1.0)
    except Exception:
        return 0.0

@lru_cache(maxsize=4096)
def _page_has_fig(path: str, mtime: float, page: int) -> bool:
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        try:
            pg = pdf[page - 1]
            try:
                return _figure_frac(pg) >= FIG_FRAC_MIN
            finally:
                pg.close()
        finally:
            pdf.close()
    except Exception:
        return False

def page_has_figure(path: Path, page: int) -> bool:
    try:
        return _page_has_fig(str(path), path.stat().st_mtime, int(page))
    except Exception:
        return False

def render_lib_page(path: Path, page: int, scale: float = 1.6):
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            pg = pdf[int(page) - 1]
            try:
                img = pg.render(scale=scale).to_pil().convert("RGB")
            finally:
                pg.close()
        finally:
            pdf.close()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
    except Exception:
        return None

def read_pages(data: bytes, name: str):
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        rd = PdfReader(io.BytesIO(data))
        if rd.is_encrypted:
            try:
                rd.decrypt("")
            except Exception:
                return []
        pages = []
        for i, pg in enumerate(rd.pages, 1):
            try:
                t = pg.extract_text() or ""
            except Exception:
                t = ""
            pages.append((i, t))
        return pages
    if ext == ".docx":
        from docx import Document
        d = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in d.paragraphs)
        for tb in d.tables:
            for row in tb.rows:
                text += "\n" + " | ".join(c.text for c in row.cells)
        return _virtual_pages(text)
    return _virtual_pages(data.decode("utf-8", errors="replace"))

def chunk_text(t: str, size: int, overlap: int):
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []
    if len(t) <= size:
        return [t] if len(t) > 80 else []
    out, i = [], 0
    while i < len(t):
        j = min(i + size, len(t))
        if j < len(t):
            k = max(t.rfind("\n", i, j), t.rfind(". ", i, j))
            if k > i + size * 0.6:
                j = k + 1
        out.append(t[i:j].strip())
        if j >= len(t):
            break
        i = max(j - overlap, i + 1)
    return [c for c in out if len(c) > 80]

def guess_year(name: str, first_text: str) -> int:
    ny = datetime.now().year
    m = [int(y) for y in YEAR_RE.findall(name) if int(y) <= ny]
    if m:
        return m[0]
    ys = [int(y) for y in YEAR_RE.findall(first_text[:3000]) if int(y) <= ny]
    return max(ys) if ys else 0

MASKS = [
    (re.compile(r"\b[1-9]\d{10}\b"), "[TCKN]"),
    (re.compile(r"(?:\+?90|0)?\s?\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b"), "[TEL]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[E-POSTA]"),
]

def mask(text: str) -> str:
    if not CFG.get("mask_pii", True):
        return text
    for rx, rep in MASKS:
        text = rx.sub(rep, text)
    return text

# =====================================================================
# 4. EMBEDDING VE VEKTÖR VERİTABANI
# =====================================================================
@st.cache_resource(show_spinner="Yerel embedding modeli yükleniyor...")
def get_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    if name in EMBED_MAX_SEQ:
        m.max_seq_length = EMBED_MAX_SEQ[name]
    return m

def _norm(vecs) -> np.ndarray:
    a = np.asarray(vecs, dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n

def _embedding_task(model_name: str, query: bool) -> str:
    if model_name.startswith("gemini:"):
        return "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
    return "LOCAL"

def _embedding_cache_key(model_name: str, task: str, text: str) -> str:
    payload = f"{model_name}\0{task}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _read_embedding_cache(keys: list[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for i in range(0, len(keys), 400):
        group = keys[i:i + 400]
        if not group:
            continue
        rows = q(
            f"SELECT cache_key, dim, emb FROM embedding_cache WHERE cache_key IN ({','.join('?' * len(group))})",
            tuple(group)
        )
        for row in rows:
            arr = np.frombuffer(row["emb"], dtype=np.float32)
            if arr.size == int(row["dim"]) and arr.size:
                found[row["cache_key"]] = arr.copy().tolist()
    return found

def _write_embedding_cache(model_name: str, task: str, texts: list[str], vectors) -> None:
    rows = []
    for text, vector in zip(texts, vectors):
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not arr.size:
            continue
        rows.append((_embedding_cache_key(model_name, task, text), model_name, task, int(arr.size), arr.tobytes(), now()))
    if rows:
        with db() as conn:
            conn.executemany("INSERT OR IGNORE INTO embedding_cache(cache_key, model, task, dim, emb, ts) VALUES (?,?,?,?,?,?)", rows)

def embed(texts: list[str], batch: int = 16, query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    name = CFG["embed_model"]
    if not name.startswith("gemini:"):
        model = get_embedder(name)
        pre = ("query: " if query else "passage: ") if "e5" in name.lower() else ""
        out: list = []
        for i in range(0, len(texts), 256):
            vecs = model.encode([pre + str(t) for t in texts[i:i + 256]], batch_size=max(int(batch), 32),
                                normalize_embeddings=True, show_progress_bar=False)
            out.extend(_norm(vecs).tolist())
        return out

    task = _embedding_task(name, query)
    prepared = [str(t)[:8000] for t in texts]
    keys = [_embedding_cache_key(name, task, t) for t in prepared]
    cached = _read_embedding_cache(list(dict.fromkeys(keys)))

    missing_by_key: dict[str, str] = {}
    for key, text in zip(keys, prepared):
        if key not in cached:
            missing_by_key.setdefault(key, text)
            
    if missing_by_key:
        missing_texts = list(missing_by_key.values())
        cli = get_cli()
        task_type = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        for i in range(0, len(missing_texts), 64):
            batch_texts = missing_texts[i:i + 64]
            r = cli.models.embed_content(
                model=name.split(":", 1)[1], contents=batch_texts,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768)
            )
            raw = [e.values for e in (r.embeddings or [])]
            vectors = _norm(raw).tolist()
            _write_embedding_cache(name, task, batch_texts, vectors)
            for t, v in zip(batch_texts, vectors):
                cached[_embedding_cache_key(name, task, t)] = v

    return [cached[key] for key in keys]

@st.cache_resource
def _mat_cache() -> dict:
    return {}

def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")[-40:].strip("_")

class VecColl:
    def __init__(self, name: str):
        self.name = name

    def count(self) -> int:
        return q("SELECT COUNT(*) AS n FROM vec WHERE coll=?", (self.name,), one=True)["n"]

    def matching_embeddings(self, ids, documents, source: str):
        if not ids:
            return []
        rows = q(f"SELECT id, doc, emb FROM vec WHERE coll=? AND src=? AND id IN ({','.join('?' * len(ids))})",
                 (self.name, source, *ids))
        by_id = {r["id"]: r for r in rows}
        matched = []
        for chunk_id, text in zip(ids, documents):
            row = by_id.get(str(chunk_id))
            if row and row["doc"] == text:
                matched.append(np.frombuffer(row["emb"], dtype=np.float32).copy())
            else:
                matched.append(None)
        return matched

    def add(self, ids, documents, embeddings, metadatas=None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        rows = [(self.name, str(i), d, json.dumps(m, ensure_ascii=False), m.get("source"),
                 np.asarray(e, dtype=np.float32).tobytes())
                for i, d, e, m in zip(ids, documents, embeddings, metadatas)]
        with db() as c:
            c.executemany("INSERT OR REPLACE INTO vec(coll, id, doc, meta, src, emb) VALUES (?,?,?,?,?,?)", rows)
        _mat_cache().pop(self.name, None)

    def delete(self, ids=None, where=None) -> None:
        with db() as c:
            if ids:
                c.executemany("DELETE FROM vec WHERE coll=? AND id=?", [(self.name, str(i)) for i in ids])
            elif where and "source" in where:
                c.execute("DELETE FROM vec WHERE coll=? AND src=?", (self.name, where["source"]))
        _mat_cache().pop(self.name, None)

    def _load(self):
        cache = _mat_cache()
        if self.name not in cache:
            rows = q("SELECT id, emb FROM vec WHERE coll=? ORDER BY rowid", (self.name,))
            ids = [r["id"] for r in rows]
            M = (np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
                 if rows else np.zeros((0, 1), dtype=np.float32))
            cache[self.name] = (ids, M)
        return cache[self.name]

    def query(self, query_embeddings, n_results: int = 10, include=None) -> dict:
        ids, M = self._load()
        empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        if not ids:
            return empty
        qv = np.asarray(query_embeddings[0], dtype=np.float32)
        sims = M @ qv
        k = max(1, min(n_results, len(ids)))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        sel = [ids[i] for i in top]
        rows = {r["id"]: r for r in q(f"SELECT id, doc, meta FROM vec WHERE coll=? AND id IN ({','.join('?' * len(sel))})",
                                     (self.name, *sel))}
        out_ids, docs, metas, dists = [], [], [], []
        for i, idx in zip(sel, top):
            r = rows.get(i)
            if not r:
                continue
            out_ids.append(i)
            docs.append(r["doc"])
            metas.append(json.loads(r["meta"] or "{}"))
            dists.append(float(1.0 - sims[idx]))
        return {"ids": [out_ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

    def get(self, limit=None, offset: int = 0) -> dict:
        if limit is None:
            return {"ids": [r["id"] for r in q("SELECT id FROM vec WHERE coll=? ORDER BY rowid", (self.name,))]}
        rows = q("SELECT id, doc, meta FROM vec WHERE coll=? ORDER BY rowid LIMIT ? OFFSET ?", (self.name, limit, offset))
        return {"ids": [r["id"] for r in rows], "documents": [r["doc"] for r in rows],
                "metadatas": [json.loads(r["meta"] or "{}") for r in rows]}

def coll(kind: str) -> VecColl:
    return VecColl(f"{kind}_{_slug(CFG['embed_model'])}")

def lib_files() -> list[Path]:
    return [p for p in sorted(LIB_DIR.iterdir())
            if p.is_file() and p.suffix.lower() in LIB_EXT
            and p.name != ARCH_FILE.name and not p.name.startswith("doc_temp_")]

def index_library(force: bool = False, progress=None) -> list[dict]:
    files = lib_files()
    c = coll("lib")
    emb_name = CFG["embed_model"]
    known = {r["name"]: r for r in q("SELECT * FROM indexed_files WHERE embed=?", (emb_name,))}
    present = {p.name for p in files}
    for n in list(known):
        if n not in present:
            c.delete(where={"source": n})
            x("DELETE FROM indexed_files WHERE name=? AND embed=?", (n, emb_name))
    report = []
    size, ov = chunk_params()

    for fi, p in enumerate(files):
        if progress:
            progress(fi / max(1, len(files)), f"{p.name} ({fi + 1}/{len(files)})")
        data = p.read_bytes()
        sha = hashlib.sha1(data).hexdigest()
        r = known.get(p.name)
        if r and r["sha1"] == sha and not force:
            actual_chunks = q("SELECT COUNT(*) AS n FROM vec WHERE coll=? AND src=?", (c.name, p.name), one=True)["n"]
            if int(actual_chunks) == int(r["chunks"]):
                report.append({"dosya": p.name, "durum": "atlandı (değişmemiş)", "parça": r["chunks"]})
                continue
        try:
            pages = read_pages(data, p.name)
        except Exception as e:
            report.append({"dosya": p.name, "durum": f"okunamadı: {e}", "parça": 0})
            continue

        empty = sum(1 for _, t in pages if len(t.strip()) < 30)
        year = guess_year(p.name, pages[0][1] if pages else "")
        pid = hashlib.md5(p.name.encode()).hexdigest()[:8]
        ids, docs, metas = [], [], []
        for pg, t in pages:
            for k, ch in enumerate(chunk_text(t, size, ov)):
                ids.append(f"{pid}:{pg}:{k}")
                docs.append(ch)
                metas.append({"source": p.name, "page": pg, "year": year})

        stage_name = f"stage_{c.name}_{pid}_{sha[:12]}"
        with db() as conn:
            conn.execute("DELETE FROM vec WHERE coll=?", (stage_name,))
        stage = VecColl(stage_name)
        committed = False
        try:
            for i in range(0, len(ids), 48):
                batch_ids = ids[i:i + 48]
                batch_docs = docs[i:i + 48]
                batch_metas = metas[i:i + 48]
                existing = c.matching_embeddings(batch_ids, batch_docs, p.name)
                vectors: list = [None] * len(batch_ids)
                missing_pos = [j for j, v in enumerate(existing) if v is None]
                if missing_pos:
                    new_vectors = embed([batch_docs[j] for j in missing_pos])
                    for j, vector in zip(missing_pos, new_vectors):
                        vectors[j] = vector
                for j, vector in enumerate(existing):
                    if vector is not None:
                        vectors[j] = vector
                stage.add(ids=batch_ids, documents=batch_docs, embeddings=vectors, metadatas=batch_metas)

            with db() as conn:
                conn.execute("DELETE FROM vec WHERE coll=? AND src=?", (c.name, p.name))
                conn.execute("INSERT OR REPLACE INTO vec(coll, id, doc, meta, src, emb) SELECT ?, id, doc, meta, src, emb FROM vec WHERE coll=?", (c.name, stage_name))
                conn.execute("DELETE FROM vec WHERE coll=?", (stage_name,))
                conn.execute("INSERT OR REPLACE INTO indexed_files(name, embed, sha1, chunks, empty_pages, total_pages, year, ts) VALUES (?,?,?,?,?,?,?,?)",
                             (p.name, emb_name, sha, len(ids), empty, len(pages), year, now()))
            committed = True
            _mat_cache().pop(c.name, None)
            report.append({"dosya": p.name, "durum": "indekslendi", "parça": len(ids)})
        finally:
            if not committed:
                with db() as conn:
                    conn.execute("DELETE FROM vec WHERE coll=?", (stage_name,))
            _mat_cache().pop(stage_name, None)

    if progress:
        progress(1.0, "Bitti")
    return report

def search_library(query: str, k: int, min_sim: float, per_file: int = 4) -> list[dict]:
    c = coll("lib")
    n = c.count()
    if n == 0:
        return []
    res = c.query(query_embeddings=[embed([query], query=True)[0]], n_results=min(k * 3, n), include=["documents", "metadatas", "distances"])
    out, per = [], {}
    for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        if sim < min_sim or per.get(m["source"], 0) >= per_file:
            continue
        per[m["source"]] = per.get(m["source"], 0) + 1
        out.append({"text": d, "source": m["source"], "page": m["page"], "year": m.get("year", 0), "sim": sim})
        if len(out) >= k:
            break
    return out

def add_insight(text: str, kind: str) -> int:
    iid = x("INSERT INTO insights(ts, kind, text) VALUES (?,?,?)", (now(), kind, text))
    coll("ins").add(ids=[str(iid)], documents=[text], embeddings=embed([text]))
    return iid

def search_insights(query: str, k: int = 4) -> list:
    c = coll("ins")
    if c.count() == 0:
        return []
    res = c.query(query_embeddings=[embed([query], query=True)[0]], n_results=min(k, c.count()))
    ids = [int(i) for i in res["ids"][0]]
    if not ids:
        return []
    rows = q(f"SELECT * FROM insights WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))
    order = {i: n for n, i in enumerate(ids)}
    return sorted(rows, key=lambda r: order[r["id"]])

def search_claims(query: str, k: int = 6, min_sim: float = 0.35) -> list[dict]:
    c = coll("clm")
    if c.count() == 0:
        return []
    res = c.query(query_embeddings=[embed([query], query=True)[0]], n_results=min(k, c.count()), include=["distances"])
    out = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        if 1 - dist < min_sim:
            continue
        r = q("SELECT * FROM claims WHERE id=?", (int(cid),), one=True)
        if r:
            out.append(dict(r))
    return out

# =====================================================================
# 5. GEMİNİ İSTEMCİSİ & İSTEK MOTORU
# =====================================================================
class QuotaError(Exception):
    pass

@st.cache_resource
def get_client(key: str):
    return genai.Client(api_key=key)

def get_cli():
    return get_client(API_KEY)

def model_for(kind: str) -> str:
    primary = CFG.get("pro_model") if kind == "pro" else CFG.get("fast_model")
    return primary or "gemini-3.8-flash"

def thinking_cfg(level: str | None, model: str):
    if not level or level == "otomatik":
        return None
    try:
        if re.search(r"gemini-([3-9])", model or ""):
            return types.ThinkingConfig(thinking_level="low" if level in ("kapalı", "düşük") else "high")
    except Exception:
        return None
    return None

def _cfg(model, system=None, temperature=None, tools=None, thinking=None, json_mode=False, drop_thinking=False):
    kw = {}
    if system:
        kw["system_instruction"] = system
    if temperature is not None:
        kw["temperature"] = temperature
    if tools:
        kw["tools"] = tools
    if json_mode:
        kw["response_mime_type"] = "application/json"
    if thinking and not drop_thinking:
        tc = thinking_cfg(thinking, model)
        if tc is not None:
            kw["thinking_config"] = tc
    return types.GenerateContentConfig(**kw)

def gen_text(contents, model: str, system=None, temperature=0.3, tools=None, thinking=None, json_mode=False, retries: int = 4) -> str:
    cli = get_cli()
    drop = False
    for i in range(retries):
        try:
            r = cli.models.generate_content(
                model=model, contents=contents,
                config=_cfg(model, system, temperature, tools, thinking, json_mode, drop)
            )
            return r.text or ""
        except Exception as e:
            if "thinking" in str(e).lower() and not drop:
                drop = True
                continue
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 ** i + 1)
                continue
            raise e
    raise QuotaError("Kota veya sunucu yoğunluğu nedeniyle istek tamamlanamadı.")

def gen_stream(contents, model: str, system, temperature, tools, thinking, meta: dict, retries: int = 4):
    cli = get_cli()
    drop = False
    for i in range(retries):
        started = False
        try:
            for ch in cli.models.generate_content_stream(
                model=model, contents=contents,
                config=_cfg(model, system, temperature, tools, thinking, False, drop)
            ):
                try:
                    t = ch.text
                except Exception:
                    t = None
                if t:
                    started = True
                    yield t
            return
        except Exception as e:
            if started:
                raise e
            if "thinking" in str(e).lower() and not drop:
                drop = True
                continue
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 ** i + 1)
                continue
            raise e
    raise QuotaError("Kota veya sunucu yoğunluğu nedeniyle istek tamamlanamadı.")

def tee(gen, buf: list):
    for t in gen:
        buf.append(t)
        yield t

def parse_json(t: str):
    t = (t or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", t, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None

# =====================================================================
# 6. SİSTEM TALİMATI (7 KATMANLI MİMARİ v3)
# =====================================================================
DEFAULT_LAYERS = """
7 KATMANLI KLİNİK AKIL YÜRÜTME ŞEMASI (iç muhakemede orantılı çalıştır):
1. VERİ, ACİLİYET VE PROBLEM TEMSİLİ: Bulgu sınıflandırması, acil durumlar (ABC, status, inme, herniasyon vb.) ve problemi sıkıştırma.
2. AYIRICI TANI (LOKALİZASYON + BAYES + FİZYOPATOLOJİ): 3 eksenli tanı (en olası, en tehlikeli, alternatifi değiştiren), Bayesçi olasılık ve mekanizma/emerjan hipotez.
3. ÇELİŞKİ VE KARŞI-HİPOTEZ ANALİZİ: Tez, Antitez (çelişen bulgular), ayırt ettirici gözlem ve sentez.
4. EN DEĞERLİ SONRAKİ ADIM VE ZAMAN EKSENİ: Bilgi değeri, karar eşiği, zamansal pencereler (0-6sa, 6-24sa vb.) ve "klinik kararı en çok iyileştiren tek değerlendirme".
5. FARMAKOLOJİK GÜVENLİK: İlaç kontrol listesi, doz güvenilirliği, bağlama özgü riskler ve seçenek karşılaştırması.
6. KENDİNİ SINAMA (KARŞI-OLGUSAL + ADVERSARİAL): Karşı-olgusal senaryolar, bilişsel yanlılık taraması (çapalama, erken kapanma vb.), çürütme testi ve kalibrasyon.
7. KARAR SENTEZİ, SİSTEM ETKİLEŞİMİ VE MENTORLUK: Çok disiplinli bakış, somut karar/gerekçe/değiştirici bulgu, acil sıra ve Sokratik mentorluk.
"""

CORE_PROMPT = """
Sen "NeroProf AI": nöroloji, nöroyoğun bakım, klinik nörofizyoloji, nöroimmünoloji, nörovasküler tıp ve nörofarmakoloji
alanlarında nörolog meslektaşa yardım eden üst düzey bir klinik karar-destek sistemisin. Amaç maksimum kesinlik taslamak değil;
verilerden en güçlü açıklamayı çıkarmak, ölümcül alternatifleri kaçırmamak, belirsizliği azaltacak en değerli adımı
bulmak ve yanlışlanabilir, kanıta dayalı kararlar önermektir. Nihai karar her zaman hekime aittir.

KAYNAK DİSİPLİNİ
- Yalnızca bağlamda verilen kaynak etiketlerine atıf yap: [K#] kütüphane, [P#] PubMed, [B#] vaka belgesi, [D#] iddia defteri.
- Etiket, PMID, DOI veya sayfa UYDURMA. Etiketsiz bilgiyi "genel bilgi (kaynaksız)" diye işaretle.
- DÜŞÜNCE DEFTERİ girdileri model hipotezleridir, kanıt değildir.
- Doz, etkileşim ve eşik değerlerinde kaynağa dayan; emin değilsen belirt.

ESKİ-YENİ KANIT
- Kaynakların yıllarını karşılaştır. YENİ ve ESKİ etiketli kaynaklar arasındaki fark ve çelişkileri göster.
- İddia defterinde "çelişkili" veya "eski" etiketli kayıtlar varsa dikkate al.

GÖRSELLER
- Bağlamda kaynak sayfasının görüntüsü ya da kullanıcı görseli varsa şekli yorumla.
- BT/MR/EEG gibi klinik görüntülerde kesin tanı koyma; gözlemleri betimle, hekim/radyolog doğrulaması öner.

GENEL
- Acil/ölümcül ayırıcı tanıları ("kaçırılmaması gerekenler") vurgula. Türkçe yanıt ver.
"""

FORMAT_COMPACT = """
YANIT BİÇİMİ (kompakt; 7 katman iç muhakemede çalışır):
## 1. Özet, aciliyet ve bulgu sınıflandırması
## 2. Ayırıcı tanı (3 eksenli: en olası, en tehlikeli, alternatifi değiştiren)
## 3. En yüksek bilgi değerli sonraki adım ve zamansal seyir
## 4. Tedavi ve farmakolojik güvenlik
## 5. Eski-yeni kanıt ve çelişkiler
## 6. Karşı-olgusal, adversarial kontrol ve belirsizlikler
## 7. Somut Karar (Karar / Gerekçe / Değiştirici Bulgu) ve Sokratik sorular
## Kaynaklar
"""

FORMAT_LAYERS = """
YANIT BİÇİMİ: 7 katmanı numaralı başlıklar altında sırayla, her biri kısa ve vakaya özgü yaz;
sonra "Eski-yeni kanıt ve çelişkiler" ile "Kaynaklar" bölümlerini ekle.
"""

def system_prompt(show_layers: bool) -> str:
    if ARCH_FILE.exists():
        arch = "MİMARİ BELGESİ (7 Katmanlı v3 - Öncelikli):\n" + ARCH_FILE.read_text(encoding="utf-8", errors="ignore")[:ARCH_CHAR_CAP]
    else:
        arch = DEFAULT_LAYERS
    return CORE_PROMPT + "\n" + arch + "\n" + (FORMAT_LAYERS if show_layers else FORMAT_COMPACT)

# =====================================================================
# 7. PUBMED
# =====================================================================
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def _eparams(extra: dict) -> dict:
    p = {"tool": "neroprof", "retmode": "json"}
    if CFG.get("ncbi_email"):
        p["email"] = CFG["ncbi_email"]
    if CFG.get("ncbi_key"):
        p["api_key"] = CFG["ncbi_key"]
    p.update(extra)
    return p

def _eget(endpoint: str, params: dict):
    r = requests.get(EUTILS + endpoint, params=params, timeout=25, headers={"User-Agent": "NeroProf/3.0"})
    r.raise_for_status()
    return r

def pubmed_ids(term: str, n: int, mindate: str | None = None, maxdate: str | None = None) -> list[str]:
    p = _eparams({"db": "pubmed", "term": term, "retmax": n, "sort": "relevance"})
    if mindate or maxdate:
        p.update(datetype="pdat", mindate=mindate or "1900/01/01", maxdate=maxdate or datetime.now().strftime("%Y/%m/%d"))
    j = _eget("esearch.fcgi", p).json()
    return j.get("esearchresult", {}).get("idlist", [])

def evidence_label(ptypes: list[str]) -> str:
    s = " ".join(ptypes).lower()
    for key, lab in (("meta-analysis", "Meta-analiz"), ("systematic review", "Sistematik derleme"),
                     ("guideline", "Kılavuz"), ("randomized controlled trial", "RKÇ"),
                     ("clinical trial", "Klinik çalışma"), ("case reports", "Olgu sunumu"),
                     ("review", "Derleme")):
        if key in s:
            return lab
    return "Diğer/belirsiz"

def pubmed_fetch(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    r = _eget("efetch.fcgi", _eparams({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
    root = ET.fromstring(r.content)
    arts = []
    for pa in root.findall(".//PubmedArticle"):
        pmid = pa.findtext(".//MedlineCitation/PMID") or ""
        te = pa.find(".//ArticleTitle")
        title = "".join(te.itertext()).strip() if te is not None else ""
        parts = []
        for at in pa.findall(".//Abstract/AbstractText"):
            lab = at.get("Label")
            txt = "".join(at.itertext()).strip()
            parts.append(f"{lab}: {txt}" if lab else txt)
        year = (pa.findtext(".//JournalIssue/PubDate/Year") or pa.findtext(".//ArticleDate/Year") or "")
        if not year:
            ym = YEAR_RE.search(pa.findtext(".//JournalIssue/PubDate/MedlineDate") or "")
            year = ym.group(1) if ym else "0"
        ptypes = [pt.text for pt in pa.findall(".//PublicationTypeList/PublicationType") if pt.text]
        arts.append({"pmid": pmid, "title": title, "abstract": " ".join(parts),
                     "journal": pa.findtext(".//Journal/Title") or "", "year": int(year) if year.isdigit() else 0,
                     "evidence": evidence_label(ptypes)})
    return arts

@st.cache_data(ttl=3600, show_spinner=False)
def pubmed_research(terms: tuple, n_recent: int, n_old: int, years: int) -> list[dict]:
    cutoff_year = datetime.now().year - years + 1
    today = datetime.now().strftime("%Y/%m/%d")
    rec_ids, old_ids = [], []
    for t in terms:
        if n_recent:
            rec_ids.extend(pubmed_ids(t, n_recent, mindate=f"{cutoff_year}/01/01", maxdate=today))
        if n_old:
            old_ids.extend(pubmed_ids(t, n_old, maxdate=f"{cutoff_year - 1}/12/31"))
    
    rec_ids, old_ids = list(dict.fromkeys(rec_ids))[:n_recent], list(dict.fromkeys(old_ids))[:n_old]
    all_ids = rec_ids + [i for i in old_ids if i not in rec_ids]
    arts = {a["pmid"]: a for a in pubmed_fetch(all_ids)}
    
    out = []
    for pid in all_ids:
        a = arts.get(pid)
        if a:
            a = dict(a)
            a["era"] = "YENİ" if a["year"] >= cutoff_year else "ESKİ"
            out.append(a)
    return out

def make_pubmed_queries(user_input: str, history: str) -> list[str]:
    p = f"Aşağıdaki klinik vaka için PubMed'de aranacak 2 İNGİLİZCE sorgu üret. Yalnızca JSON: {{\"queries\": [\"...\", \"...\"]}}\n\nSORU: {user_input[:2000]}"
    try:
        out = parse_json(gen_text(p, model_for("fast"), temperature=0, json_mode=True, thinking="kapalı"))
        qs = out.get("queries", []) if isinstance(out, dict) else []
        qs = [s.strip() for s in qs if isinstance(s, str) and s.strip()][:2]
    except Exception:
        qs = []
    return qs or [re.sub(r"\s+", " ", user_input)[:150]]

# =====================================================================
# 8. İNANÇ DEFTERİ (İDDİA VE ÇELİŞKİ YÖNETİMİ)
# =====================================================================
CLAIM_PROMPT = """Aşağıdaki metinden klinik açıdan önemli, doğrulanabilir iddiaları çıkar.
Çıktı yalnızca JSON dizisi:
[{"claim": "tek cümlelik iddia", "topic": "kısa konu", "year": 2024 veya 0,
  "evidence": "Meta-analiz|RKÇ|Kılavuz|Derleme|Olgu|Uzman görüşü|Belirsiz",
  "source_ref": "[K2] veya [P3] gibi bağlam etiketi, yoksa 'model'"}]
"""

def ingest_claims(answer_text: str, sources_desc: str) -> tuple[int, int]:
    fast = model_for("fast")
    prompt = CLAIM_PROMPT + f"\nETİKET LİSTESİ:\n{sources_desc[:3000]}\n\nMETİN:\n{answer_text[:12000]}"
    items = parse_json(gen_text(prompt, fast, temperature=0, json_mode=True, thinking="kapalı"))
    if not isinstance(items, list):
        return 0, 0
    items = [i for i in items if isinstance(i, dict) and len(str(i.get("claim", ""))) > 15][:8]
    if not items:
        return 0, 0
    cc = coll("clm")
    vecs = embed([str(i["claim"]) for i in items])
    n_new = 0
    for it, v in zip(items, vecs):
        yr = int(it.get("year") or 0)
        ref = str(it.get("source_ref", "model"))
        stype = "kütüphane" if ref.startswith("[K") else "pubmed" if ref.startswith("[P") else "model"
        cid = x("INSERT INTO claims(ts, claim, topic, source_type, source_ref, year, evidence, status, note) VALUES (?,?,?,?,?,?,?,?,?)",
                (now(), str(it["claim"]), str(it.get("topic", "")), stype, ref, yr, str(it.get("evidence", "Belirsiz")), "aktif", ""))
        cc.add(ids=[str(cid)], documents=[str(it["claim"])], embeddings=[v])
        n_new += 1
    return n_new, 0

# =====================================================================
# 9. BAĞLAM İNŞASI
# =====================================================================
def build_context(user_input: str, case_id: int, o: dict):
    sources, parts, warnings = [], [], []
    today = datetime.now().strftime("%Y-%m-%d")
    cutoff_year = datetime.now().year - int(o["recent_years"]) + 1

    prev = q("SELECT role, content FROM messages WHERE case_id=? ORDER BY id DESC LIMIT 9", (case_id,))[1:]
    prev = list(reversed(prev))
    hist_text = "\n".join(f"{r['role']}: {r['content'][:1200]}" for r in prev)
    rq = user_input

    # Vaka belgeleri
    docs = q("SELECT * FROM case_docs WHERE case_id=? ORDER BY id", (case_id,))
    budget, blocks = DOC_CHAR_CAP, []
    for i, d in enumerate(docs, 1):
        if budget <= 0:
            break
        t = d["text"][:budget]
        budget -= len(t)
        blocks.append(f"[B{i}] {d['name']}\n{t}")
        sources.append({"label": f"B{i}", "title": d["name"], "detail": "vaka belgesi"})
    parts.append("## VAKA BELGELERİ\n" + ("\n\n".join(blocks) if blocks else "(yok)"))

    # Kütüphane RAG
    lib_blocks, fig_parts, fig_labels = [], [], []
    if o["use_lib"]:
        try:
            seen_pages = set()
            for i, h in enumerate(search_library(rq, o["k_lib"], o["min_sim"]), 1):
                yr = f"yıl≈{h['year']}" if h["year"] else "yıl bilinmiyor"
                era = " YENİ" if h["year"] and h["year"] >= cutoff_year else " ESKİ"
                lib_blocks.append(f"[K{i}] {h['source']} | s.{h['page']} | {yr}{era}\n{h['text']}")
                src_item = {"label": f"K{i}", "title": f"{h['source']} s.{h['page']}", "detail": f"{yr}, benzerlik {h['sim']:.2f}", "snippet": h["text"][:300]}
                key = (h["source"], h["page"])
                if (o.get("send_figs") and len(seen_pages) < int(o.get("max_figs", 2)) and key not in seen_pages and h["source"].lower().endswith(".pdf")):
                    path = LIB_DIR / h["source"]
                    if page_has_figure(path, h["page"]) or FIG_RE.search(h["text"]):
                        jpg = render_lib_page(path, h["page"])
                        if jpg:
                            seen_pages.add(key)
                            fig_parts += [f"[K{i}] kaynağının sayfa görüntüsü ({h['source']} s.{h['page']}):", types.Part.from_bytes(data=jpg, mime_type="image/jpeg")]
                            fig_labels.append(f"K{i} s.{h['page']}")
                sources.append(src_item)
        except Exception as e:
            warnings.append(f"Kütüphane araması başarısız: {e}")
    parts.append("## KÜTÜPHANE ALINTILARI\n" + ("\n\n".join(lib_blocks) if lib_blocks else "(ilgili alıntı yok)"))

    # PubMed
    pm_blocks = []
    if o["pubmed"]:
        try:
            terms = tuple(make_pubmed_queries(user_input, hist_text))
            arts = pubmed_research(terms, int(o["n_recent"]), int(o["n_old"]), int(o["recent_years"]))
            for i, a in enumerate(arts, 1):
                pm_blocks.append(f"[P{i}] PMID {a['pmid']} | {a['year']} | {a['era']} | {a['evidence']} | {a['journal']}\n{a['title']}\n{a['abstract'][:1800] or '(özet yok)'}")
                sources.append({"label": f"P{i}", "title": f"{a['title'][:110]}", "detail": f"{a['year']} {a['era']} · {a['evidence']}", "url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"})
        except Exception as e:
            warnings.append(f"PubMed erişilemedi: {e}")
    parts.append("## PUBMED (güncel literatür)\n" + ("\n\n".join(pm_blocks) if pm_blocks else "(kullanılmadı/yok)"))

    # Hafıza
    claim_lines, ins_lines = [], []
    if o["use_mem"]:
        try:
            for i, c in enumerate(search_claims(rq, 6), 1):
                claim_lines.append(f"[D{i}] ({c['status']}) {c['source_type']}/{c['source_ref']} yıl {c['year'] or '?'} · {c['evidence']}: {c['claim']}")
                sources.append({"label": f"D{i}", "title": c["claim"][:110], "detail": f"{c['status']} · {c['year'] or '?'}"})
            for r in search_insights(rq, 4):
                ins_lines.append(f"- ({r['ts']}, {r['kind']}) {r['text'][:700]}")
        except Exception as e:
            warnings.append(f"Hafıza araması başarısız: {e}")
    parts.append("## İNANÇ DEFTERİ\n" + ("\n".join(claim_lines) or "(yok)"))
    parts.append("## DÜŞÜNCE DEFTERİ (MODEL HİPOTEZLERİ)\n" + ("\n".join(ins_lines) or "(yok)"))
    parts.append("## GEÇMİŞ KONUŞMA\n" + (hist_text or "(yok)"))
    parts.append(f"## YENİ SORU / VAKA\n{user_input}")

    header = f"# BUGÜN: {today}\n# 'YENİ' kaynak = {cutoff_year} ve sonrası; 'ESKİ' = öncesi.\n"
    return header + "\n\n".join(parts), sources, warnings, fig_parts

CRITIC_PROMPT = """Aşağıdaki BAĞLAM ve CEVAP'ı denetle. 7 Katmanlı Klinik Akıl Yürütme Mimarisine göre "Denetçi Notu" yaz:
1) Uydurma/bağlamda olmayan atıf veya doz.
2) Kaçırılmış acil/ölümcül tanı.
3) Doz, etkileşim, organ fonksiyonu hataları.
4) Erken kapanma veya sahte kesinlik.
Sorun yoksa "Belirgin sorun saptanmadı" yaz.
"""

# =====================================================================
# 10. VAKA YÖNETİMİ
# =====================================================================
def new_case(title: str | None = None) -> int:
    title = title or f"Vaka {datetime.now().strftime('%d.%m %H:%M')}"
    return x("INSERT INTO cases(title, created, archived) VALUES (?,?,0)", (title, now()))

def add_message(case_id: int, role: str, content: str, sources=None) -> int:
    return x("INSERT INTO messages(case_id, role, content, sources, ts) VALUES (?,?,?,?,?)",
             (case_id, role, content, json.dumps(sources or [], ensure_ascii=False), now()))

def cb_new_case():
    t = (st.session_state.get("new_case_title") or "").strip()
    st.session_state.case_sel = new_case(t or None)
    st.session_state.new_case_title = ""

def cb_archive_case():
    cid = st.session_state.case_sel
    cur = q("SELECT archived FROM cases WHERE id=?", (cid,), one=True)
    if cur and not cur["archived"]:
        x("UPDATE cases SET archived=1 WHERE id=?", (cid,))
        st.session_state.case_sel = new_case()
    else:
        x("UPDATE cases SET archived=0 WHERE id=?", (cid,))

# =====================================================================
# 11. YAN PANEL
# =====================================================================
def sidebar() -> None:
    sb = st.sidebar
    sb.markdown("## 🧠 NeroProf AI v3")

    with sb.expander("🔑 API & Modeller", expanded=not API_KEY):
        with st.form("cfg_form"):
            key_in = st.text_input("Gemini API anahtarı", type="password", placeholder="(kayıtlı)" if CFG.get("api_key") else "AIza...")
            fast_m = st.text_input("Hızlı Model", value=CFG.get("fast_model", "gemini-3.8-flash"))
            pro_m = st.text_input("Güçlü Model", value=CFG.get("pro_model", "gemini-3.8-flash"))
            emb_labels = list(EMBED_MODELS)
            cur_lab = next((k for k, v in EMBED_MODELS.items() if v == CFG["embed_model"]), emb_labels[0])
            emb_lab = st.selectbox("Embedding modeli (yerel)", emb_labels, index=emb_labels.index(cur_lab))
            mask_pii = st.checkbox("TCKN/telefon/e-postayı otomatik maskele", value=CFG.get("mask_pii", True))
            
            if st.form_submit_button("💾 Kaydet"):
                if key_in.strip():
                    CFG["api_key"] = key_in.strip()
                CFG["fast_model"] = fast_m.strip() or "gemini-3.8-flash"
                CFG["pro_model"] = pro_m.strip() or "gemini-3.8-flash"
                CFG["embed_model"] = EMBED_MODELS[emb_lab]
                CFG["mask_pii"] = mask_pii
                save_cfg(CFG)
                st.rerun()

    if not API_KEY:
        return

    with sb.expander("⚙️ Çalışma modu & araçlar", expanded=True):
        st.radio("Mod", list(MODES), index=1, key="mode", horizontal=True)
        st.checkbox("📚 Kütüphane (RAG)", value=True, key="o_lib")
        st.checkbox("🧠 Hafıza (iddia + düşünce defteri)", value=True, key="o_mem")
        st.checkbox("🔬 PubMed (güncel literatür)", value=False, key="o_pubmed")
        st.checkbox("🌍 Google Search", value=False, key="o_web")
        st.checkbox("🧩 7 katmanı açık yaz", value=False, key="o_layers")
        st.checkbox("🖼️ Şekilli kaynak sayfalarının görüntüsünü modele ekle", value=True, key="o_figs")
        st.slider("En fazla sayfa görüntüsü", 1, 4, 2, key="max_figs")
        st.checkbox("⚖️ Yanıttan iddiaları deftere işle", value=False, key="o_claims")
        st.slider("Kütüphane benzerlik eşiği", 0.0, 0.8, 0.30, 0.05, key="min_sim")
        st.slider("PubMed: YENİ makale sayısı", 0, 10, 6, key="n_recent")
        st.slider("PubMed: ESKİ makale sayısı", 0, 10, 4, key="n_old")
        st.slider("'YENİ' = son kaç yıl", 1, 10, 3, key="recent_years")

    with sb.expander("📁 Vaka yönetimi", expanded=True):
        cases = q("SELECT * FROM cases ORDER BY archived, id DESC")
        ids = [c["id"] for c in cases]
        labels = {c["id"]: f"#{c['id']} {c['title']}{' 🗄️' if c['archived'] else ''}" for c in cases}
        if st.session_state.get("case_sel") not in ids:
            st.session_state.case_sel = ids[0]
        st.selectbox("Aktif vaka", ids, format_func=lambda i: labels[i], key="case_sel")
        st.text_input("Yeni vaka başlığı", key="new_case_title", placeholder="65y E, akut iskemik inme")
        st.button("➕ Yeni vaka aç", on_click=cb_new_case)
        st.button("🗄️ Arşivle / arşivden çıkar", on_click=cb_archive_case)

def get_opts() -> dict:
    m = MODES[st.session_state.get("mode", "Standart")]
    ss = st.session_state
    return {
        **m,
        "use_lib": ss.get("o_lib", True), "use_mem": ss.get("o_mem", True),
        "pubmed": ss.get("o_pubmed", False), "web": ss.get("o_web", False),
        "layers": ss.get("o_layers", False), "claims": m["claims"] or ss.get("o_claims", False),
        "min_sim": ss.get("min_sim", 0.30), "n_recent": ss.get("n_recent", 6),
        "n_old": ss.get("n_old", 4), "recent_years": ss.get("recent_years", 3),
        "send_figs": ss.get("o_figs", True), "max_figs": ss.get("max_figs", 2),
    }

# =====================================================================
# 12. GÖRÜNÜMLER
# =====================================================================
def render_sources(srcs: list) -> None:
    with st.expander(f"📎 Kullanılan kaynaklar ({len(srcs)})"):
        for s in srcs:
            line = f"**[{s['label']}]** {s.get('title', '')}"
            if s.get("detail"):
                line += f" — _{s['detail']}_"
            if s.get("url"):
                line += f"  [bağlantı]({s['url']})"
            st.markdown(line)

def render_msg(m) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        try:
            srcs = json.loads(m["sources"] or "[]")
        except Exception:
            srcs = []
        if srcs:
            render_sources(srcs)

def handle_message(prompt: str, uploads) -> None:
    case_id = st.session_state.case_sel
    o = get_opts()
    user_text = mask(prompt)
    imgs, attached = [], []
    for f in uploads or []:
        ext = Path(f.name).suffix.lower()
        data = f.getvalue()
        if ext in IMG_EXT:
            imgs.append(types.Part.from_bytes(data=data, mime_type=IMG_EXT[ext]))
            attached.append(f"{f.name} (görüntü)")
        else:
            try:
                txt = mask("\n".join(t for _, t in read_pages(data, f.name)))
            except Exception as e:
                st.warning(f"{f.name} okunamadı: {e}")
                continue
            if txt.strip():
                x("INSERT INTO case_docs(case_id, name, text, ts) VALUES (?,?,?,?)", (case_id, f.name, txt, now()))
                attached.append(f.name)

    st.session_state.up_n += 1
    user_content = user_text + (f"\n\n📎 Ekler: {', '.join(attached)}" if attached else "")
    add_message(case_id, "user", user_content)
    with st.chat_message("user"):
        st.markdown(user_content)

    with st.chat_message("assistant"):
        buf, meta, sources, answer, err = [], {}, [], "", None
        try:
            with st.status("Kaynaklar toplanıyor...", expanded=False) as stt:
                ctx, sources, warns, fig_parts = build_context(user_text, case_id, o)
                stt.update(label=f"Bağlam hazır ({len(sources)} kaynak)", state="complete")
            for w in warns:
                st.warning(w)
            model = model_for(o["model"])
            tools = [types.Tool(google_search=types.GoogleSearch())] if o["web"] else None
            gen = gen_stream([ctx] + fig_parts + imgs, model, system_prompt(o["layers"]), 0.4, tools, o["think"], meta)
            answer = st.write_stream(tee(gen, buf)) or ""
        except QuotaError as e:
            err = f"⚠️ {e}"
        except Exception as e:
            err = f"⚠️ Hata: {e}"

        if err:
            answer = "".join(buf) + ("\n\n" if buf else "") + err
            st.markdown(err)

        if o["critic"] and not err:
            try:
                with st.spinner("🛡️ Denetçi geçişi..."):
                    note = gen_text(f"{CRITIC_PROMPT}\n\nBAĞLAM:\n{ctx[:90000]}\n\nCEVAP:\n{answer}",
                                    model_for("pro"), temperature=0.1, thinking="orta")
                if note.strip():
                    block = f"\n\n---\n**🛡️ Denetçi Notu**\n\n{note.strip()}"
                    st.markdown(block)
                    answer += block
            except Exception as e:
                st.warning(f"Denetçi geçişi başarısız: {e}")

        if sources:
            render_sources(sources)
        add_message(case_id, "assistant", answer, sources)

        if o["claims"] and not err:
            try:
                desc = "\n".join(f"[{s['label']}] {s.get('title', '')}" for s in sources)
                n_new, _ = ingest_claims(answer, desc)
                st.toast(f"İnanç defterine {n_new} iddia işlendi.")
            except Exception as e:
                st.toast(f"İddia işleme başarısız: {e}")
    st.rerun()

def view_chat() -> None:
    case_id = st.session_state.case_sel
    case = q("SELECT * FROM cases WHERE id=?", (case_id,), one=True)
    st.subheader(f"💡 {case['title']}" + (" 🗄️ (arşivde)" if case["archived"] else ""))
    docs = q("SELECT id, name FROM case_docs WHERE case_id=? ORDER BY id", (case_id,))
    if docs:
        with st.expander(f"📎 Bu vakaya ekli belgeler ({len(docs)})"):
            for d in docs:
                c1, c2 = st.columns([8, 1])
                c1.write(d["name"])
                if c2.button("🗑️", key=f"deldoc_{d['id']}"):
                    x("DELETE FROM case_docs WHERE id=?", (d["id"],))
                    st.rerun()
    uploads = st.file_uploader(
        "Mesaja ekle: PDF/DOCX/TXT/MD veya görüntü",
        type=[e.strip(".") for e in LIB_EXT] + [e.strip(".") for e in IMG_EXT],
        accept_multiple_files=True, key=f"up_{st.session_state.up_n}"
    )
    for m in q("SELECT * FROM messages WHERE case_id=? ORDER BY id", (case_id,)):
        render_msg(m)
    prompt = st.chat_input("Vaka detayı, bulgular, laboratuvar değerleri veya klinik soru...")
    if prompt:
        handle_message(prompt, uploads)

def view_library() -> None:
    st.subheader("📚 Kütüphane (Yerel RAG)")
    if ARCH_FILE.exists():
        st.success(f"omni_neuro_architecture.md (7 Katmanlı v3) aktif ({ARCH_FILE.stat().st_size:,} bayt).")
    else:
        st.info("kutuphane/omni_neuro_architecture.md bulunamadı; yerleşik 7 katman tanımı kullanılıyor.")

    ups = st.file_uploader("Kütüphaneye dosya ekle", type=[e.strip(".") for e in LIB_EXT], accept_multiple_files=True, key=f"libup_{st.session_state.libup_n}")
    if ups and st.button("⬇️ Yüklenenleri kütüphaneye kaydet"):
        for f in ups:
            (LIB_DIR / Path(f.name).name).write_bytes(f.getvalue())
        st.session_state.libup_n += 1
        st.rerun()

    known = {r["name"]: r for r in q("SELECT * FROM indexed_files WHERE embed=?", (CFG["embed_model"],))}
    rows = []
    for p in lib_files():
        r = known.get(p.name)
        rows.append({"dosya": p.name, "durum": "✅ indeksli" if r else "⏳ bekliyor", "parça": r["chunks"] if r else 0})
    if rows:
        st.dataframe(rows, use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("🔄 Yeni/değişen dosyaları indeksle"):
        prog = st.progress(0.0, text="Başlıyor...")
        index_library(force=False, progress=lambda f, t: prog.progress(min(f, 1.0), text=t))
        st.success("İndeksleme tamamlandı.")
        st.rerun()
    if c2.button("♻️ Tümünü yeniden indeksle"):
        prog = st.progress(0.0, text="Başlıyor...")
        index_library(force=True, progress=lambda f, t: prog.progress(min(f, 1.0), text=t))
        st.success("Yeniden indeksleme tamamlandı.")
        st.rerun()

def view_thoughts() -> None:
    st.subheader("🧠 Düşünce Defteri")
    st.caption("Burası MODEL HİPOTEZLERİDİR; kaynak değildir.")
    rows = q("SELECT * FROM insights ORDER BY id DESC LIMIT 40")
    for r in rows:
        with st.expander(f"#{r['id']} · {r['ts']} · {r['kind']}"):
            st.markdown(r["text"])

def view_ledger() -> None:
    st.subheader("⚖️ İnanç Defteri: İddialar ve Çelişkiler")
    rows = q("SELECT * FROM claims ORDER BY id DESC LIMIT 200")
    st.dataframe([{"id": r["id"], "durum": r["status"], "iddia": r["claim"], "yıl": r["year"] or "?", "kanıt": r["evidence"], "kaynak": f"{r['source_type']} {r['source_ref']}"} for r in rows], use_container_width=True)

def view_reports() -> None:
    st.subheader("📋 Klinik Raporlama")
    st.info("Sistematik epikriz ve SOAP devirlik notu üretimi aktif vakaya göre yapılır.")

# =====================================================================
# 13. ANA AKIŞ
# =====================================================================
if not q("SELECT id FROM cases LIMIT 1"):
    new_case("Vaka 1")

sidebar()
st.markdown("# 🧠 NeroProf AI v3")
st.caption("7 Katmanlı Klinik Akıl Yürütme Mimarisi · Yerel RAG · PubMed · Kalıcı Hafıza")

if not API_KEY:
    st.warning("Başlamak için sol panelden Gemini API anahtarınızı girip kaydedin.")
    st.stop()

st.session_state.setdefault("up_n", 0)
st.session_state.setdefault("libup_n", 0)

VIEWS = ["💬 Konsültasyon", "📚 Kütüphane", "🧠 Düşünce Defteri", "⚖️ İnanç Defteri", "📋 Raporlar"]
view = st.radio("Görünüm", VIEWS, horizontal=True, label_visibility="collapsed", key="view")

if view == VIEWS[0]:
    view_chat()
elif view == VIEWS[1]:
    view_library()
elif view == VIEWS[2]:
    view_thoughts()
elif view == VIEWS[3]:
    view_ledger()
else:
    view_reports()
