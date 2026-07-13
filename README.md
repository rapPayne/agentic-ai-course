# Agentic AI for Python — 1-Day Intensive for Experienced Developers

A fast-paced, lecture-only day for developers with existing programming experience. The focus is the Python/AI slice of a larger full-stack capstone: just enough Python, FastAPI, LangChain, and LangGraph to build the inference server for the **Intelligent Onboarding Assistant** capstone described in [INSTRUCTORNOTES.md](INSTRUCTORNOTES.md), [PHASE1.md](PHASE1.md), and [PHASE2.md](PHASE2.md). There are no labs — every chapter is a lecture that feeds directly into capstone project time.

---

## Prerequisites

- Comfortable writing code in at least one programming language — no prior Python assumed
- Basic understanding of HTTP, REST, and JSON
- Familiarity with using an LLM conversationally (ChatGPT, Claude, Copilot) — no ML or AI background needed
- Git and a terminal you're not afraid of
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) installed before the session

This session covers only the Python/AI inference server. React, Node/Express, MongoDB, and JWT-based auth — the rest of the capstone's stack — are assumed to come from elsewhere. They appear here only as the systems the inference server talks to, never as topics taught in this course.

---

## Time Breakdown

**Total instruction time: ~5 hours**, excluding breaks.

The day runs 9:00–5:00. Breaks: 15 min morning, 60 min lunch, 15 min afternoon = 90 min, leaving 6h30min available. Five hours of lecture leaves roughly 90 minutes of buffer for Q&A and capstone kickoff — deliberately generous, since this is new material for most students and there are no labs to absorb the usual overflow.

### Content Summary

| # | Type | Topic | Duration |
|---|------|-------|----------|
| 1 | Lecture | Python Basics for Experienced Developers | 60 min |
| 2 | Lecture | FastAPI | 45 min |
| 3 | Lecture | LangChain Fundamentals | 60 min |
| 4 | Lecture | Retrieval-Augmented Generation (RAG) | 45 min |
| 5 | Lecture | LangGraph: Stateful & Agentic Workflows | 60 min |
| 6 | Lecture | Responsible AI & Agentic Guardrails | 30 min |

---

## What This Course Does Not Cover

This is deliberately narrow — one day, no labs, straight into the capstone:

- **React, Node/Express, MongoDB** — the other two-thirds of the capstone's stack, taught elsewhere
- **Real-time streaming responses** — an explicit "extra-mile" idea in `INSTRUCTORNOTES.md`, not core content
- **LangGraph checkpointing/persistence** — the capstone's inference server is stateless per request by design; each call carries its own context from the database API
- **Deployment, containers, or CI/CD** — running the inference server locally is enough

---

## Additional Resources

- [LangChain documentation](https://python.langchain.com/)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Chroma documentation](https://docs.trychroma.com/)
- [uv documentation](https://docs.astral.sh/uv/)

---

Copyright © 2026 Agile Gadgets, LLC. All rights reserved. See [COPYRIGHT.md](COPYRIGHT.md).
