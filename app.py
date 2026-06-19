'''
app.py

VASP RAG chat interface using chainlit

Run:
    chainlit run app.py

    GEMINI_API_KEY = ...
    GEMINI_MODEL   = gemini-2.0-flash-lite  (optional)
    WIKI_TOP_K     = 2                       (optional)
    FORUM_TOP_K    = 3                       (optional)
'''

import os
from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from retriever import retrieve, format_context
from llm import stream_response

WIKI_TOP_K  = int(os.getenv('WIKI_TOP_K',  '2'))
FORUM_TOP_K = int(os.getenv('FORUM_TOP_K', '3'))


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            '**VASP Assistant** ready.\n\n'
            'Ask your VASP question here'
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    if not query:
        return

    # ── 1. Retrieve ───────────────────────────────────────────────
    chunks = retrieve(query, wiki_top_k=WIKI_TOP_K, forum_top_k=FORUM_TOP_K)

    if not chunks:
        await cl.Message(
            content='No relevant context found. Try rephrasing your question.'
        ).send()
        return

    context = format_context(chunks)

    # ── 2. Stream LLM response ────────────────────────────────────
    response_msg = cl.Message(content='')
    await response_msg.send()

    try:
        for token in stream_response(query, context):
            await response_msg.stream_token(token)
    except Exception as e:
        await response_msg.update()
        await cl.Message(
            content=f'**Error:** {e}\n\nCheck your `GEMINI_API_KEY` in `.env`.'
        ).send()
        return

    await response_msg.update()

    # ── 3. Sources panel ──────────────────────────────────────────
    source_lines = []
    for i, chunk in enumerate(chunks, 1):
        label = f'[{chunk.source.upper()}]'
        match = '**Keyword Match**' if chunk.match_type == 'keyword' else ''
        url   = chunk.url or 'no url'
        source_lines.append(
            f'{i}. {label}{match} **{chunk.title}** (score: {chunk.score:.3f})  \n'
            f'   {url}'
        )

    await cl.Message(
        content=f'**Sources ({len(chunks)}):**\n\n' + '\n'.join(source_lines),
        parent_id=response_msg.id,
    ).send()