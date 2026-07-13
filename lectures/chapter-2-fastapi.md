# Chapter 2: FastAPI

## Overview

FastAPI maps roughly to Express with two differences that matter immediately: request/response bodies are typed `Pydantic` models instead of raw `req.body`/`res.json()` objects, and validation happens automatically from those type annotations instead of by hand in the handler. This chapter covers enough to build simple POST routes that can call another HTTP service. FastAPI's WebSocket support and streaming responses are out of scope for this course.

## A Minimal App

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

```bash
uv run uvicorn main:app --reload --port 8000
```

`uvicorn` is the ASGI server that actually runs a FastAPI app — FastAPI just defines routes, it doesn't listen on a socket itself. `main:app` means "the `app` object in `main.py`," so it has to match your filename and variable name.

Every running FastAPI app serves an interactive Swagger UI at `/docs` for free — no client setup required to try a route by hand.

## Routes and Path Parameters

```python
@app.get("/users/{user_id}/tasks")
async def get_tasks(user_id: str) -> list[dict]:
    return await load_tasks(user_id)
```

A path param typed in the function signature is validated and coerced automatically — a `user_id: int` route rejects a non-numeric segment with a 422 before your code runs, unlike Express's `req.params`, which is always a string you coerce yourself.

## Request Bodies as Pydantic Models

```python
from pydantic import BaseModel

class AskRequest(BaseModel):
    role: str
    department: str
    question: str

@app.post("/ask")
async def ask(body: AskRequest) -> dict:
    answer = await answer_question(body.role, body.department, body.question)
    return {"answer": answer}
```

FastAPI parses the request body, validates every field against `AskRequest`, and rejects malformed input with a 422 — before `ask`'s body ever runs. There's no equivalent of forgetting to check `req.body.question` for `undefined`.

## Response Models

```python
class AskResponse(BaseModel):
    answer: str
    citation: str | None = None

@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    ...
    return AskResponse(answer=answer, citation=citation)
```

Returning a `BaseModel` instance serializes it to JSON automatically — no manual `res.json(...)` call.

## Pydantic Field Types, Briefly

A field can be any type hint: a plain type (`str`, `int`, `float`, `bool`), a generic (`list[str]`, `dict[str, int]`), or another `BaseModel` nested directly:

```python
class Task(BaseModel):
    text: str
    completed: bool = False

class OnboardingPlan(BaseModel):
    role: str
    tasks: list[Task]          # nested model
```

Make a field optional/nullable with `| None`, plus a default so the caller can omit it:

```python
class AskResponse(BaseModel):
    citation: str | None = None   # optional — omit or pass null
```

Without `= None`, `str | None` is still required (just nullable) — the caller must send the key, even if its value is `null`.

## Async Route Handlers

Every route that calls an LLM or another service is `async def`, and awaits that call inside the handler body:

```python
@app.post("/plan")
async def plan(body: AskRequest) -> dict:
    result = await onboarding_chain.ainvoke({"role": body.role, "department": body.department})
    return result
```

## Dependency Injection, Briefly

`Depends()` lets a route declare a shared piece of setup once and reuse it across handlers:

```python
from fastapi import Depends, Header

async def get_bearer_token(authorization: str = Header(...)) -> str:
    return authorization.removeprefix("Bearer ")

@app.post("/ask")
async def ask(body: AskRequest, token: str = Depends(get_bearer_token)) -> dict:
    ...
```

## Reading and Forwarding the `Authorization` Header

The inference server never issues its own JWT — it receives one from the React front end on every request and reuses it when a tool needs to call the database API on the user's behalf.

```python
from fastapi import Header

@app.post("/ask")
async def ask(body: AskRequest, authorization: str = Header(...)) -> dict:
    jwt = authorization.removeprefix("Bearer ")
    ...
```

## Calling Another HTTP Service

```python
import httpx

async def add_task(user_id: str, text: str, jwt: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:3001/users/{user_id}/tasks",
            json={"text": text},
            headers={"Authorization": f"Bearer {jwt}"},
        )
        response.raise_for_status()
        return response.json()
```

This is the pattern a tool one one server uses to hit an API on another server, forwarding the JWT it received from the front end rather than authenticating as itself.

## CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Same problem, same one-line fix as Express's `cors()` package — a React dev server and an inference server run on different ports which the browser treats as different origins.

## Error Handling

```python
from fastapi import HTTPException

@app.get("/users/{user_id}/tasks")
async def get_tasks(user_id: str) -> list[dict]:
    tasks = await load_tasks(user_id)
    if tasks is None:
        raise HTTPException(status_code=404, detail="No tasks found")
    return tasks
```

An uncaught exception anywhere in a handler returns a generic 500 by default. `HTTPException` is how you return a deliberate status code and message instead. Unlike Express, where a response is something you construct and send (`res.status(404).json(...)`), FastAPI catches the raised exception for you and turns it into the HTTP response — `raise` is the mechanism for sending a non-200 response, not just an error signal.

## Environment Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["ANTHROPIC_API_KEY"]
```

## Gotchas

- A Pydantic validation failure returns **422**, not 400 — a different status-code vocabulary than an Express background trains you to expect.
- An absent `Authorization` header doesn't reject a request by itself unless the handler declares it as a required parameter (as in `Header(...)`) — otherwise it's just `None`, and the code that needed it fails somewhere further down, off in the weeds.
- Calling a chain's sync `.invoke()` inside an `async def` route works but blocks the event loop for every other in-flight request being served by the same process — always await the async variant inside a route handler.
- `httpx.AsyncClient()` is cheap to use inside a `with` block per call for this course's scale, but in a hot path it's normally reused across requests rather than instantiated fresh every time — worth knowing, not worth over-engineering here.

## Quick Reference

```python
# Minimal app
app = FastAPI()

@app.get("/path/{id}")
async def handler(id: int) -> dict: ...

@app.post("/path")
async def handler(body: SomeModel) -> SomeModel: ...

# Header + Depends
async def get_token(authorization: str = Header(...)) -> str: ...
@app.post("/path")
async def handler(token: str = Depends(get_token)): ...

# httpx call with forwarded auth
async with httpx.AsyncClient() as client:
    r = await client.post(url, json=payload, headers={"Authorization": f"Bearer {jwt}"})

# CORS
app.add_middleware(CORSMiddleware, allow_origins=[...], allow_methods=["*"], allow_headers=["*"])

# Errors
raise HTTPException(status_code=404, detail="...")

# Run
uv run uvicorn main:app --reload --port 8000
```

| Status code | When |
|---|---|
| 200 | success |
| 201 | resource created |
| 404 | not found (raised manually) |
| 422 | Pydantic validation failure (automatic) |
| 500 | uncaught exception (automatic) |

## Coming Up

Once the LangChain chapter adds the chain that goes inside the handler body — the line where `answer_question` and `onboarding_chain.ainvoke` are still stubs above — these routes have a real chain behind them instead of a stub.
