# Chapter 1: Python Basics for Experienced Developers

## Overview

Python is dynamically typed, interpreted, and indentation-sensitive — the syntax will be the biggest adjustment, not the concepts. This chapter is a bridge, not a Python course: it covers exactly the subset of the language the rest of this course leans on — `uv` for running code, type hints, core data structures, functions, `Pydantic.BaseModel`, control flow, `async`/`await`, error handling, modules, and string formatting. Idiomatic Python well outside that set — metaclasses, multiple inheritance, decorator authoring, `**kwargs` gymnastics — is out of scope for this course.

## Running Code with `uv`

`uv` replaces `pip` + `venv` + `pip-tools` with one tool. It creates and manages the virtual environment for you — you never activate one by hand.

```bash
uv init onboarding-inference       # scaffold a project
cd onboarding-inference
uv add fastapi langchain langgraph # adds to pyproject.toml, creates/updates the venv
uv run main.py                     # runs inside the project's venv, no activation step
```

`uv add` is the equivalent of `npm install <package>` — it writes the dependency to `pyproject.toml` the same way `npm install` writes to `package.json`. `uv run` is the equivalent of `npm run <script>` combined with automatically making sure `node_modules` is up to date first.

## Type Hints on a Dynamically Typed Language

Python doesn't check types at runtime — a type hint is documentation and editor tooling, not enforcement. This course writes them anyway, on every function, because FastAPI and Pydantic use them to build real runtime validation later.

```python
def greeting_for(role: str, department: str) -> str:
    return f"Welcome to {department}, {role}."
```

Nothing stops you from calling `greeting_for(1, None)` — Python will happily try and fail at runtime inside the function body instead of rejecting the call up front. The editor's language server (via `pyright` or similar) is what flags the mismatch as you type; there's no compiler in the loop.

## Core Data Structures

```python
tasks: list[str] = ["Set up laptop", "Meet your manager", "Complete I-9"]
new_hire: dict[str, str] = {"role": "Software Engineer", "department": "Engineering"}
window = ("plan30Day", "plan60Day", "plan90Day")   # tuple — fixed-size, usually heterogeneous
seen_roles: set[str] = {"Software Engineer", "Payroll Specialist"}
```

Comprehensions are the idiomatic map/filter — reach for them before a manual loop:

```python
open_tasks = [t for t in tasks if not t.startswith("Completed:")]
by_role = {hire["role"]: hire["department"] for hire in new_hires}
```

## Functions

```python
def build_task(text: str, completed: bool = False) -> dict:
    return {"text": text, "completed": completed}

build_task("Set up laptop")
build_task("Meet your manager", completed=True)

def summarize(role: str, *tasks: str) -> str:          # *args — recognize, rarely write
    return f"{role}: {len(tasks)} tasks"

as_upper = lambda text: text.upper()                     # prefer a named def over this for anything non-trivial
```

## `Pydantic.BaseModel` as the Default Data Shape

Python has classes, but this course doesn't hand-write them for plain data. Every data shape — a request body, a response body, structured LLM output — is a `Pydantic.BaseModel` subclass instead: typed fields, nothing else. You'll see this exact pattern reused, unchanged, in the FastAPI and LangChain chapters.

```python
from pydantic import BaseModel

class NewHire(BaseModel):
    role: str
    department: str
    day_in_onboarding: int = 0   # default value, same idea as a Python function default
```

## Control Flow

```python
if new_hire["role"] == "Manager":
    view = "team"
elif day_in_onboarding <= 30:
    view = "plan30Day"
else:
    view = "dashboard"

for task in tasks:            # for...of equivalent — iterates values
    print(task)

for key in new_hire:          # for...in equivalent — iterates dict keys
    print(key, new_hire[key])

with open("policy.txt") as f:  # context manager — guaranteed cleanup, no explicit finally
    text = f.read()
```

## Async/Await

`async def` marks a function as a coroutine; `await` pauses it and yields control back to the event loop rather than blocking the process. Python's asyncio loop is single-threaded and cooperative — the same mental model as Node's event loop.

```python
async def fetch_policy_chunks(query: str) -> list[str]:
    results = await vector_store.asimilarity_search(query)
    return [r.page_content for r in results]
```

## Error Handling

```python
class LowConfidenceError(Exception):
    pass

try:
    answer = await generate_answer(question)
except LowConfidenceError:
    answer = "I don't have a confident answer to that."
finally:
    log_event("ask_answered")

raise LowConfidenceError("no supporting chunks found")  # `throw` equivalent
```

## Imports and Modules

```python
from models import NewHire            # named import, same file layout convention as ESM
import chains                          # module import, access via chains.build_ask_chain(...)
from .tools import manage_task         # relative import within a package
```

## f-Strings

f-strings are the only string-formatting form used in this course — the direct analog to JS template literals.

```python
name = "Software Engineer"
message = f"Generating a 30/60/90 plan for a {name} in {new_hire['department']}."
```

## `None` and Falsiness

```python
plan30 = new_hire.get("plan30Day")
if plan30 is None:              # idiomatic — not `== None`
    plan30 = "Not yet generated."

if not tasks:                   # empty list is falsy, same idea as JS []
    print("No tasks yet.")
```

## Gotchas

- **Indentation is syntax, not style.** A mismatched indent level is a `SyntaxError`, or worse, silently changes which block a line belongs to — there are no braces to fall back on.
- **`is None`, not `== None`.** Idiomatic, and occasionally not equivalent once a class overrides `__eq__`.
- **Mutable default arguments are a classic trap.** The default is created once, at function-definition time, and shared across every call that doesn't pass its own:

  ```python
  def add_task(task: str, existing: list[str] = []) -> list[str]:  # bug
      existing.append(task)
      return existing
  ```

  Every call that omits `existing` mutates the *same* list. Use `None` and create the list inside the function body instead.
- **Calling an `async def` function without `await` doesn't raise an error.** It returns a coroutine object, not the result — the same silent trap as an un-awaited Promise in JavaScript.
- **f-strings only.** `.format()` and `%`-formatting exist in the language but should not appear in code you write for this course.

## Quick Reference

| Form | Syntax |
|---|---|
| Type hints | `def f(x: int, y: str = "a") -> bool:` |
| List/dict comprehension | `[x for x in xs if cond]`, `{k: v for k, v in items}` |
| Function, default arg | `def f(x, y=0): ...` |
| Function, keyword-only call | `f(x=1, y=2)` |
| `*args` / `**kwargs` | `def f(*args, **kwargs): ...` |
| Lambda | `lambda x: x * 2` |
| Pydantic model | `class M(BaseModel): field: str` |
| `async`/`await` | `async def f(): ... ; await g()` |
| `try`/`except`/`finally` | `try: ... except SomeError: ... finally: ...` |
| f-string | `f"{value}"` |

**`uv` commands:** `uv init`, `uv add <pkg>`, `uv add --dev <pkg>`, `uv run <file>`, `uv sync`.

## Coming Up

The FastAPI chapter's route handlers and request bodies are a direct application of the type hints and `BaseModel` syntax just covered — the exact same `BaseModel` shape you just wrote for `NewHire` becomes the request body for `POST /ask`.
