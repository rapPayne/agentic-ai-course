# Chapter 4: Retrieval-Augmented Generation (RAG)

## Overview

An LLM's training data doesn't include ADP's private HR policy documents, so when we ask it to answer produces a hallucination. RAG fixes this by retrieving the actual relevant text at answer time and feeding it into the prompt as grounding context. This chapter builds the full pipeline: load, chunk, embed, store, retrieve, cite. We'll use Chroma as our vector DB, and an embedding model is treated as a black box with an `.embed_query()` interface.

The first four steps — load, chunk, embed, store — are **ingestion**: run once, offline, before anyone asks a question. The last two — retrieve, cite — are **inference**: run fresh on every incoming request. The two run as separate processes and only share one thing on disk: `persist_directory`.

## Ingestion - Document Loading

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Full-time employees accrue 1.25 PTO days per month, capped at 15 days in year one.",
    metadata={"source": "ADP_PTO_Policy.txt"},
)
```

`metadata` is where the source document's name lives. Capture it now, at load time — the vector store has no idea which file a chunk came from unless you attach that yourself.

## Ingestion - Loading Different File Formats

A source document is rarely a hand-built string — it's a PDF, a DOCX, an HTML export, or a plain `.txt` file on disk. A loader reads the real file and produces the same `Document` shape shown above, `metadata` included, so everything downstream (chunking, embedding, storage) doesn't care which format it started as:

```bash
uv add pypdf docx2txt beautifulsoup4
```

```python
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, BSHTMLLoader, TextLoader

docs = PyPDFLoader("policy.pdf").load()      # list[Document], one per page
docs = Docx2txtLoader("policy.docx").load()  # list[Document], one for the whole file
docs = BSHTMLLoader("policy.html").load()    # list[Document], tags stripped
docs = TextLoader("policy.txt").load()       # list[Document], read as-is
```

## Ingestion - Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents([doc])
```

`chunk_overlap` exists because a fact split exactly at a chunk boundary (ie. the PTO accrual rate in one chunk, the year-one cap in the next) can become unretrievable as a single coherent answer. Overlap means a boundary rarely cuts cleanly through a sentence that matters.

## Ingestion - Embeddings

An embedding model turns text into a vector where similar meaning ends up close together in that vector space. Instantiate a client; there's no need to understand the underlying math to use one correctly.

```python
from langchain_anthropic import AnthropicEmbeddings   # or your provider's embedding client

embeddings = AnthropicEmbeddings(model="voyage-3")
```

It doesn't need to be from the same provider as the chat model — but whichever one you pick has to embed every query the same way it embedded the documents, so switching later means re-embedding the whole store from scratch.

## Ingestion - Vector Storage with Chroma

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_store",
)
```

Chroma runs embedded in your process — unlike PostgreSQL, there's no separate server to install or connect to. `persist_directory` is what makes the store survive a process restart; without it, everything you just ingested is gone the next time you run the script.

## Inference - Reconnecting to a Persisted Store

**Inference starts here.** Everything above this point runs once, offline, as ingestion. Everything from here on runs on every incoming request, in a separate process — one that never calls `.from_documents()` again, because the store already exists on disk:

```python
embeddings = AnthropicEmbeddings(model="voyage-3")
vector_store = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)
```

No `documents=` argument, and nothing gets re-embedded — this just opens the store `.from_documents(...)` already built and wrote to `persist_directory`. It still needs the same `embeddings` object as ingestion did, since every incoming query has to be embedded the same way the stored documents were.

## Inference - Retrieval

```python
results = vector_store.similarity_search("How much PTO do I get in my first year?", k=3)

for r in results:
    print(r.metadata["source"], r.page_content[:80])
```

`k` is how many chunks to return — `k=3` means the 3 closest matches. Each result still carries the `metadata` attached at load time — this is what makes a citation possible later.

## Inference - Similarity Scores and Confidence Gating

```python
results_with_scores = vector_store.similarity_search_with_score(question, k=3)

CONFIDENCE_THRESHOLD = 0.75

(best_doc, best_score) = results_with_scores[0]
if best_score < CONFIDENCE_THRESHOLD:
    answer = "I don't have a confident answer to that in our policy documents."
else:
    answer = await generate_grounded_answer(question, [d for (d, _) in results_with_scores])
```

`similarity_search` on its own always returns *something*, even when nothing in the store is actually relevant. Checking the threshold helps protect against hallucinations.

## Inference - Building the Grounded, Cited Answer

```python
from langchain_core.prompts import ChatPromptTemplate

grounded_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the policy text below. Policy text:\n{context}"),
    ("human", "{question}"),
])

async def generate_grounded_answer(question: str, docs: list[Document]) -> dict:
    context = "\n\n".join(d.page_content for d in docs)
    chain = grounded_prompt | llm | parser
    answer_text = await chain.ainvoke({"context": context, "question": question})
    sources = sorted({d.metadata["source"] for d in docs})
    return {"answer": answer_text, "citation": sources}
```

The citation is built from `d.metadata["source"]` — never from asking the model which document it used. This cites everything the model *saw*, not necessarily everything it *used*: a deliberate over-citation, since there's no reliable way to know which retrieved chunks the answer actually drew from.

## Gotchas

- Retrieval quality depends entirely on chunking, not on the model. A chunk that splits a fact from its qualifying sentence retrieves a misleading half-answer.
- `similarity_search` always returns results, even against an empty or irrelevant query — silence isn't a possible outcome. We should have a score threshold you check yourself.
- A Chroma store created without `persist_directory` disappears when the process exits, silently forcing a full re-ingest the next run.
- Citation text must come from stored metadata because a model asked "which document did you use?" answers plausibly, not necessarily correctly.
- An embedding model has its own max input length, separate from the chat model's context window. A chunk sized for the LLM but too long for the embedding model gets silently truncated before embedding, quietly degrading retrieval with no error raised.

## Quick Reference

```python
# Document
Document(page_content="...", metadata={"source": "filename.txt"})

# Loaders — each returns list[Document]
PyPDFLoader("file.pdf").load()
Docx2txtLoader("file.docx").load()
BSHTMLLoader("file.html").load()
TextLoader("file.txt").load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Vector store — ingestion (embeds and writes to disk)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_store")

# Vector store — inference (reopens the existing store, no re-embedding)
vector_store = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

# Retrieval
vector_store.similarity_search(query, k=3)
vector_store.similarity_search_with_score(query, k=3)   # (Document, score) pairs

# Citation from metadata
sources = {d.metadata["source"] for d in retrieved_docs}
```

## Coming Up

This retrieve-and-cite pipeline becomes a single node — `retrieve` — in the graph the LangGraph chapter builds next, sitting between `gather` and `generate` in a gather → retrieve → generate → adapt sequence. The confidence gate above gets revisited explicitly in the Responsible AI chapter, as a checkable field on the response rather than a hedge buried in the answer text.
