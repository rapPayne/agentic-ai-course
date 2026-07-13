# Chapter 3: LangChain Fundamentals

## Overview

LangChain is not a framework that hides the LLM call — it's a thin, consistent interface over different providers plus a small set of composable pieces: prompts, models, and output parsers. This chapter builds the core pattern the rest of this course leans on: a prompt template piped into a chat model piped into a parser. Agents, multi-step orchestration, and retrieval are out of scope here — those are the RAG and LangGraph chapters.

## Chat Models

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatAnthropic(model="claude-sonnet-5")

response = llm.invoke([
    SystemMessage(content="You are an onboarding assistant for ADP."),
    HumanMessage(content="How much PTO do I get?"),
])
print(response.content)
```

`SystemMessage`, `HumanMessage`, and `AIMessage` are the three message types you'll construct by hand or see returned from a chain.

## Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an onboarding assistant. The user is a {role} in {department}."),
    ("human", "{question}"),
])

rendered = prompt.invoke({"role": "Software Engineer", "department": "Engineering", "question": "What tools do I need on day one?"})
```

`{variable}` placeholders are filled by the dict passed to `.invoke()` — this is the mechanism that turns role/department context into a personalized prompt.

## Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()   # plain text out
```

For structured data — a typed task list instead of a free-text blob you'd have to hand-parse — use structured output instead of a parser stage:

```python
from pydantic import BaseModel

class TaskList(BaseModel):
    tasks: list[str]

structured_llm = llm.with_structured_output(TaskList)
```

## LCEL Composition

The pipe operator chains a prompt, a model, and a parser into one runnable — this is the current, recommended composition style. The older `LLMChain`/`SequentialChain` classes and `AgentExecutor` are legacy APIs and don't appear anywhere in this course.

```python
chain = prompt | llm | parser

result = chain.invoke({"role": "Software Engineer", "department": "Engineering", "question": "What tools do I need on day one?"})
```

Each stage's output becomes the next stage's input: the rendered prompt goes into the model, the model's response goes into the parser.

## Invoking a Chain

```python
result = chain.invoke({...})        # sync — blocks the event loop if called inside an async route
result = await chain.ainvoke({...}) # async — what a FastAPI handler should actually call
```

`.stream()` exists for token-by-token output but isn't covered in this course — streaming is an extra-mile idea, not a requirement.

## System Prompts as Context Injection

The same chain produces genuinely different output for a Payroll Specialist in Finance than a Software Engineer in Engineering, because the role and department are rendered directly into the system message on every call:

```python
plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a first-week task list for a {role} in {department} at ADP."),
    ("human", "Generate the task list now."),
])

plan_chain = plan_prompt | structured_llm

task_list = await plan_chain.ainvoke({"role": "Payroll Specialist", "department": "Finance"})
```

This is the whole mechanism behind a route like `POST /ask` or `POST /plan` accepting role/department context — there's no separate personalization step, just what's in the prompt.

## Tool Definitions with `@tool`

Decorate a plain Python function with a docstring and type-hinted parameters, and LangChain turns it into a schema a model can choose to call. Nothing in this chapter wires a tool up to anything yet — that happens in the LangGraph chapter, once there's a graph to route the decision through.

```python
from langchain_core.tools import tool

@tool
def flag_for_manager_review(user_id: str, reason: str) -> str:
    """Escalate a new hire to their manager when they appear stuck."""
    return "flagged"
```

The docstring is not documentation for humans here — it's what the model reads to decide whether this tool fits the current request.

## Gotchas

- `.invoke()` runs fine inside an `async def` route but blocks the event loop for every other request in flight on the same process — always call `.ainvoke()` from a route handler.
- A prompt template's `{variable}` placeholders must match the dict keys passed to `.invoke()` exactly — a typo isn't caught when the chain is defined, only when it's called, and the error message points at the prompt, not the typo.
- `.with_structured_output()` depends on the underlying model actually supporting structured or tool output — this isn't universal across every provider or model, so it's worth confirming for whichever one you're using.
- The `@tool` decorator's docstring is load-bearing, not decorative — a vague description produces vague tool-selection behavior once a model is actually deciding whether to call it.

## Quick Reference

```python
# Chat model
llm = ChatAnthropic(model="claude-sonnet-5")

# Messages
SystemMessage(content="...")
HumanMessage(content="...")
AIMessage(content="...")

# Prompt template
prompt = ChatPromptTemplate.from_messages([("system", "..."), ("human", "{var}")])

# Parsers / structured output
StrOutputParser()
llm.with_structured_output(SomeModel)

# LCEL
chain = prompt | llm | parser
result = await chain.ainvoke({"var": "..."})

# Tool definition
@tool
def my_tool(arg: str) -> str:
    """One sentence the model uses to decide when to call this."""
    ...
```

## Coming Up

This chain — prompt to LLM to parser — is what sits behind a FastAPI route from Chapter 2: `POST /ask` and `POST /plan` are each one chain like this. The `@tool` shape shown at the end returns in the LangGraph chapter as the mechanism behind the manage-tasks, negotiate-plan, and flag-for-review tools.
