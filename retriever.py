'''
retriever.py
============
VASP RAG retriever. keyword priority + weighted hybrid search + URL merging.

Pipeline:
  1. Keyword lookup  — wiki chunks whose title exactly matches a query term
  2. Wiki vector     — top WIKI_TOP_K by cosine similarity
  3. Forum vector    — top FORUM_TOP_K by cosine similarity
  4. Re-score        — apply source weights
  5. Dedup           — by (title, chunk_index, text[:100])
  6. Merge by URL    — chunks from same page combined into one, text concatenated in chunk_index order, score = max across group
'''

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

# Config

CHROMA_PATH     = 'chroma_storage'
COLLECTION_NAME = 'vasp_rag'
EMBED_MODEL     = 'intfloat/e5-base-v2'
MIN_CHUNK_LEN   = 100

WIKI_TOP_K   = 2
FORUM_TOP_K  = 3
KEYWORD_MAX  = 2

WIKI_WEIGHT  = 1.10
FORUM_WEIGHT = 1.00

STOPWORDS = {
    'what', 'is', 'how', 'do', 'i', 'set', 'the', 'a', 'an',
    'and', 'or', 'to', 'in', 'for', 'of', 'it', 'can', 'should',
    'when', 'why', 'which', 'does', 'my', 'me', 'use', 'used',
    'get', 'give', 'tell', 'explain', 'show', 'list',
}


# singletons

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    print(f'[retriever] Loading {EMBED_MODEL}...')
    return SentenceTransformer(EMBED_MODEL, device='cpu')


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


# Result dataclass

@dataclass
class RetrievedChunk:
    text:         str
    source:       str
    title:        str
    url:          str
    score:        float
    vector_score: float
    chunk_index:  int
    match_type:   str = 'vector'


def _dedup_key(title: str, chunk_index: int, text: str) -> tuple:
    return (title, chunk_index, text[:100])


def extract_terms(query: str) -> list[str]:
    tokens = re.findall(r'[a-zA-Z0-9_]+', query)
    return [t for t in tokens if t.lower() not in STOPWORDS]


def _is_junk(doc: str) -> bool:
    return (
        doc.strip().upper().startswith('REDIRECT')
        or len(doc.strip()) < MIN_CHUNK_LEN
    )


def _make_chunk(
    doc: str,
    meta: dict,
    vector_score: float,
    match_type: str,
    weight: float = 1.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        text         = doc,
        source       = meta.get('source', ''),
        title        = meta.get('title', ''),
        url          = meta.get('url', ''),
        score        = round(vector_score * weight, 4),
        vector_score = vector_score,
        chunk_index  = meta.get('chunk_index', 0),
        match_type   = match_type,
    )


# Keyword title lookup

def _keyword_lookup(terms: list[str]) -> list[RetrievedChunk]:
    collection = _get_collection()
    chunks: list[RetrievedChunk] = []
    seen: set[tuple] = set()

    for term in terms:
        for candidate in [term.upper(), term, term.lower()]:
            try:
                results = collection.get(
                    where={'$and': [
                        {'title':  {'$eq': candidate}},
                        {'source': {'$eq': 'wiki'}},
                    ]},
                    limit=10,
                    include=['documents', 'metadatas'],
                )
            except Exception:
                continue

            for doc, meta in zip(results['documents'], results['metadatas']):
                if _is_junk(doc):
                    continue
                key = _dedup_key(meta.get('title',''), meta.get('chunk_index',0), doc)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(_make_chunk(doc, meta, 1.0, 'keyword', WIKI_WEIGHT))

            if any(c.title == candidate for c in chunks):
                break

        if len(chunks) >= KEYWORD_MAX:
            break

    return chunks[:KEYWORD_MAX]


# Vector search

def _vector_search(
    embedding: list[float],
    source:    str,
    top_k:     int,
    seen:      set[tuple],
    weight:    float = 1.0,
) -> list[RetrievedChunk]:
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k * 10,
        where={'source': source},
        include=['documents', 'metadatas', 'distances'],
    )

    chunks = []

    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
    ):
        if _is_junk(doc):
            continue
        key = _dedup_key(meta.get('title',''), meta.get('chunk_index',0), doc)
        if key in seen:
            continue
        seen.add(key)

        vscore = round(1.0 - dist, 4)
        chunks.append(_make_chunk(doc, meta, vscore, 'vector', weight))

        if len(chunks) == top_k:
            break

    return sorted(chunks, key=lambda c: c.score, reverse=True)


# URL merging

def merge_by_url(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    '''
    Merge chunks from the same URL into a single chunk.

    - Chunks are concatenated in chunk_index order
    - Score = max score across the group
    - Result order = first-seen order (preserves ranking)
    '''
    groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
    order:  list[str] = []

    for chunk in chunks:
        key = chunk.url or chunk.title
        if key not in groups:
            order.append(key)
        groups[key].append(chunk)

    merged = []
    for key in order:
        group = sorted(groups[key], key=lambda c: c.chunk_index)
        best  = max(group, key=lambda c: c.score)

        combined_text = '\n\n---\n\n'.join(c.text for c in group)

        merged.append(RetrievedChunk(
            text         = combined_text,
            source       = best.source,
            title        = best.title,
            url          = best.url,
            score        = best.score,
            vector_score = best.vector_score,
            chunk_index  = group[0].chunk_index,
            match_type   = best.match_type,
        ))

    return merged


# Public API

def retrieve(
    query:       str,
    wiki_top_k:  int = WIKI_TOP_K,
    forum_top_k: int = FORUM_TOP_K,
) -> list[RetrievedChunk]:
    '''
    Retrieve relevant VASP chunks using keyword + vector hybrid search.
    Chunks from the same URL are merged into one before returning.

    Result order: keyword wiki → wiki vector → forum vector
    '''
    model = _get_model()
    terms = extract_terms(query)

    keyword_chunks = _keyword_lookup(terms)
    seen: set[tuple] = {
        _dedup_key(c.title, c.chunk_index, c.text)
        for c in keyword_chunks
    }

    embedding = model.encode(
        f'query: {query}',
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    wiki_chunks  = _vector_search(embedding, 'wiki',  wiki_top_k,  seen, WIKI_WEIGHT)
    forum_chunks = _vector_search(embedding, 'forum', forum_top_k, seen, FORUM_WEIGHT)

    try:
        paper_chunks = _vector_search(embedding, 'paper', 1, seen, 1.0)
    except Exception:
        paper_chunks = []

    all_chunks = keyword_chunks + wiki_chunks + forum_chunks + paper_chunks

    # Merge chunks from the same URL into one
    return merge_by_url(all_chunks)


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        tag          = 'KEYWORD MATCH — ' if chunk.match_type == 'keyword' else ''
        source_label = f'[{chunk.source.upper()}] {tag}{chunk.title}'
        parts.append(f'--- Source {i}: {source_label} ---\n{chunk.text}')
    return '\n\n'.join(parts)


# testing 

if __name__ == '__main__':
    import sys

    query = ' '.join(sys.argv[1:]) or 'What is ENCUT and how do I set it?'
    print(f'\nQuery: {query}\n')

    chunks = retrieve(query)

    if not chunks:
        print('No results — check that chroma_storage is populated.')
    else:
        for i, c in enumerate(chunks, 1):
            print(
                f'[{i}] score={c.score:.4f}  '
                f'match={c.match_type:7s}  '
                f'source={c.source}  '
                f'title={c.title}'
            )
            print(c.text[:300])
            print()