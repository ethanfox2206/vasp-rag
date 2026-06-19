'''
append.py

Add one or more PDFs to the existing vasp_rag ChromaDB collection.

Usage:
    python append.py paper1.pdf paper2.pdf ...
    python append.py papers/*.pdf

Skips PDFs already ingested (tracked in pdf_manifest.json).
Use --force to re-ingest a file even if it's in the manifest.
'''

import argparse
import hashlib
import json
import sys
import time
from itertools import islice
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Config

CHROMA_PATH     = 'chroma_storage'
COLLECTION_NAME = 'vasp_rag'
EMBED_MODEL     = 'intfloat/e5-base-v2'

EMBED_BATCH_SIZE  = 64
UPSERT_BATCH_SIZE = 2500

MAX_CHARS       = 1500
OVERLAP_CHARS   = 200
MIN_CHUNK_CHARS = 100
MIN_ADVANCE     = 256

MANIFEST_PATH = Path('pdf_manifest.json')


# Manifest

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# PDF extraction

def extract_pdf_text(path: Path) -> tuple[str, dict]:
    import pymupdf
    doc   = pymupdf.open(str(path))
    meta  = doc.metadata or {}
    pages = [page.get_text() for page in doc if page.get_text().strip()]
    doc.close()
    return '\n\n'.join(pages), {
        'title':      meta.get('title') or path.stem,
        'author':     meta.get('author', ''),
        'filename':   path.name,
        'page_count': len(pages),
    }


# Chunking

def split_by_chars(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        text = text.strip()
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    chunks = []
    start  = 0

    while start < len(text):
        end = start + MAX_CHARS

        if end >= len(text):
            chunk = text[start:].strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
            break

        split_pos = end
        for sep in ('\n\n', '\n', '. ', ' '):
            pos = text.rfind(sep, start + MIN_ADVANCE, end)
            if pos > start + MIN_ADVANCE:
                split_pos = pos
                break

        chunk = text[start:split_pos].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        start = max(split_pos - OVERLAP_CHARS, start + MIN_ADVANCE)

    return chunks


def chunk_pdf(text: str, meta: dict):
    for idx, chunk in enumerate(split_by_chars(text)):
        yield {
            'text':        chunk,
            'source':      'paper',
            'title':       meta['title'],
            'url':         meta['filename'],
            'author':      meta['author'],
            'chunk_index': idx,
        }


# Batching

def batched(iterator, size):
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Append PDFs to the vasp_rag ChromaDB collection.'
    )
    parser.add_argument(
        'pdfs',
        nargs='+',
        type=Path,
        help='One or more PDF files to ingest',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-ingest files already in the manifest',
    )
    args = parser.parse_args()

    # Validate inputs
    pdfs = []
    for p in args.pdfs:
        if not p.exists():
            print(f'  skip (not found): {p}')
        elif p.suffix.lower() != '.pdf':
            print(f'  skip (not a PDF): {p}')
        else:
            pdfs.append(p)

    if not pdfs:
        print('No valid PDF files provided.')
        sys.exit(1)

    # Check manifest
    manifest = load_manifest()
    to_ingest = []

    for pdf in pdfs:
        h = file_hash(pdf)
        if not args.force and manifest.get(str(pdf)) == h:
            print(f'  skip (already ingested): {pdf.name}  (use --force to re-ingest)')
        else:
            to_ingest.append(pdf)

    if not to_ingest:
        print('\nNothing to ingest.')
        sys.exit(0)

    print(f'\nIngesting {len(to_ingest)} file(s)...\n')

    # Load model + collection
    print(f'Loading {EMBED_MODEL}...')
    model = SentenceTransformer(EMBED_MODEL, device='cpu')

    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(
        COLLECTION_NAME,
        metadata={'hnsw:space': 'cosine'},
    )

    existing = collection.count()
    print(f'Existing chunks : {existing:,}\n')

    next_id       = existing
    total_chunks  = 0
    embed_seconds = 0
    write_seconds = 0

    buffer_ids  = []
    buffer_docs = []
    buffer_meta = []
    buffer_embs = []

    for pdf_path in to_ingest:
        print(f'→ {pdf_path.name}')

        try:
            text, meta = extract_pdf_text(pdf_path)
        except Exception as e:
            print(f'  ERROR reading file: {e}')
            continue

        if not text.strip():
            print(f'  WARNING: no text extracted (scanned PDF? needs OCR)')
            continue

        chunks = list(chunk_pdf(text, meta))
        print(f'  {len(chunks)} chunks')

        for batch in tqdm(
            batched(chunks, EMBED_BATCH_SIZE),
            desc='  embedding',
            unit='batch',
            leave=False,
        ):
            texts = [f'passage: {c['text']}' for c in batch]

            t0 = time.perf_counter()
            embeddings = model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embed_seconds += time.perf_counter() - t0

            for chunk, emb in zip(batch, embeddings):
                meta_out = {k: v for k, v in chunk.items() if k != 'text'}
                buffer_ids.append(f'paper_{next_id}')
                buffer_docs.append(chunk['text'])
                buffer_meta.append(meta_out)
                buffer_embs.append(emb.tolist())
                next_id      += 1
                total_chunks += 1

            if len(buffer_ids) >= UPSERT_BATCH_SIZE:
                t0 = time.perf_counter()
                collection.add(
                    ids=buffer_ids,
                    documents=buffer_docs,
                    metadatas=buffer_meta,
                    embeddings=buffer_embs,
                )
                write_seconds += time.perf_counter() - t0
                buffer_ids.clear()
                buffer_docs.clear()
                buffer_meta.clear()
                buffer_embs.clear()

        manifest[str(pdf_path)] = file_hash(pdf_path)
        save_manifest(manifest)
        print(f'  done ✓')

    # Flush remainder
    if buffer_ids:
        t0 = time.perf_counter()
        collection.add(
            ids=buffer_ids,
            documents=buffer_docs,
            metadatas=buffer_meta,
            embeddings=buffer_embs,
        )
        write_seconds += time.perf_counter() - t0

    print(f'\n{'─' * 40}')
    print(f'Chunks added    : {total_chunks:,}')
    print(f'Total in DB     : {collection.count():,}')
    print(f'Embedding time  : {embed_seconds:.1f}s')
    print(f'Write time      : {write_seconds:.1f}s')
    print(f'Manifest saved  : {MANIFEST_PATH}')


if __name__ == '__main__':
    main()