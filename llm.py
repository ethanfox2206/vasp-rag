'''
llm.py
======
Gemini LLM wrapper for VASP RAG.

Requires:
    pip install google-genai
    GEMINI_API_KEY in .env
'''

from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = '''
You are a helpful assistant specialising in the Vienna Ab initio Simulation
Package (VASP). Answer questions using the provided context excerpts from the
VASP wiki, VASP forum, and academic literature.

Guidelines:
- Be precise and cite the source label (e.g. [WIKI], [FORUM]) when relevant.
- If the context does not contain enough information, say so clearly rather
than guessing.
- Use correct VASP terminology (INCAR tags in ALL_CAPS, e.g. ENCUT, ISMEAR).
- Keep answers concise unless the user asks for detail.
'''


def build_prompt(query: str, context: str) -> str:
    return (
        f'Context from VASP documentation:\n\n'
        f'{context}\n\n'
        f'---\n\n'
        f'Question: {query}'
    )


def stream_response(query: str, context: str) -> Iterator[str]:
    '''Stream a Gemini response for *query* given *context*.'''
    from google import genai

    api_key = os.getenv('GEMINI_API_KEY', '')
    model   = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite')

    client   = genai.Client(api_key=api_key)
    prompt   = build_prompt(query, context)

    response = client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text