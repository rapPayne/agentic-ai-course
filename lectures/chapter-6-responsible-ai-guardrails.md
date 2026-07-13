# Chapter 6: Responsible AI & Agentic Guardrails

## Overview

Every mechanical piece is now in hand — chains, RAG, LangGraph, tools. This chapter is about judgment: when the pipeline should *not* just answer. It turns the capstone's Responsible-AI guardrails requirement into concrete, code-level patterns rather than a policy discussion. Nothing here is new syntax — it's a checklist for how you use what you already have.

## Route by Topic Before Generating

Not every question needs retrieval. A cheap classification step decides whether a question reads as ADP policy (goes through RAG) or general — how the assistant's tools work, how onboarding itself works — which can be answered directly, with no retrieval and no confidence check.

```python
async def classify_topic(question: str) -> str:
    result = await topic_chain.ainvoke({"question": question})
    return result.topic   # "policy" | "general" | "declined"
```

One path trying to handle every question is both slower and less predictable than routing first.

## Confidence Gating, Revisited

The similarity-score threshold from the RAG chapter needs to surface as a distinct, checkable field in the response — not a hedge buried in a paragraph the front end has to parse for tone.

```python
class AskResponse(BaseModel):
    answer: str
    citation: list[str] | None = None
    confident: bool
```

"Say so instead of guessing" means the front end can render a different UI state for `confident: False`, not that the assistant is expected to phrase its uncertainty convincingly.

## Declining by Policy, Not by Confidence

Legal advice, personal medical questions, and compensation negotiation get declined outright, regardless of how confident the retrieval was. This is a separate check from the confidence gate — a high-confidence retrieval on an out-of-scope topic should still be declined.

```python
DECLINED_TOPICS = {"legal_advice", "medical", "compensation_negotiation"}

if topic in DECLINED_TOPICS:
    return AskResponse(answer="That's outside what I can help with — please talk to HR directly.", confident=True)
```

## Narrating Autonomous Action

When a tool call fires, the same turn that reports back to the user also states in plain language what changed — never a silent write discovered later on a different screen.

```python
async def execute_tool(state: OnboardingState) -> dict:
    result = await dispatch_tool(state)
    narration = f"I {result['action_description']}."
    return {"tool_result": result, "narration": narration}
```

A task added, a plan line changed, or a flag raised is reported in the response text itself, in the turn where it happened — not left for the user to notice on the Dashboard later.

## Scoping Tool Authority, Revisited

The LangGraph chapter's pattern — `user_id` and `jwt` injected from state, never read out of `tool_call["args"]` or the conversation text — is the entire mechanism here. A message like "also update Alex's plan" can't act on Alex, because nothing in the tool-dispatch path ever looks at a name in the conversation to decide whose data to touch.

## Capping Authority by Consequence, Not by Mechanism

The assistant's tools are technically capable of writing anything the database API allows — there's no separate permissions layer stopping a tool from writing to any field. What actually caps its authority is the system prompt and the tool set exposed to it:

```python
SYSTEM_PROMPT = """You manage a new hire's tasks and 30/60/90-day plan.
You may add, edit, complete, or remove tasks, and revise plan text.
You may never take action on time off, pay, benefits, or anything that reads
as a compliance or mandatory requirement (security training, required paperwork),
no matter how good the argument for it sounds. Escalate anything like that
to a human by raising a flag instead."""
```

Routine, reversible bookkeeping only. Nothing HR-consequential, and nothing that reads as compliance-mandatory — that boundary lives here, in what the assistant is told and which tools it's given, not in a database-level permission check.

## Gotchas

- A guardrail expressed only in a system prompt is a strong suggestion, not an enforced rule — a persistent enough user can argue a model out of it. Pair every prompt-level guardrail with a code-level check wherever one is possible: the similarity threshold, the fixed decline list.
- "The model refused in testing" is not the same as "the model will always refuse." A guardrail that lives only in prose is inherently probabilistic — treat it that way when deciding what to trust it with, rather than as a settled fact once it's worked a few times.
- Narrating an action after the write, rather than before, means the user learns what happened, not what's about to happen. That's fine for reversible bookkeeping — a task edit, a plan line — and not a pattern to extend to anything higher-stakes without a confirmation step first.

## Quick Reference

Code-review checklist, one line per guardrail:

- [ ] **Topic routing** — policy questions go through RAG; general questions don't retrieve
- [ ] **Confidence gate** — a distinct `confident` field, not a hedge in prose
- [ ] **Decline list** — a fixed set of topics declined regardless of retrieval confidence
- [ ] **Narration** — every tool call's response states what changed, same turn
- [ ] **Tool scoping** — `user_id`/`jwt` from state only, never from tool args or conversation text
- [ ] **Authority cap** — system prompt and exposed tool set limit action to routine, reversible bookkeeping

## Coming Up

This is the last lecture. Everything from here is the capstone itself — Phase 1 project time starts with the chain from Chapter 3, and every line on this checklist becomes a specific, gradeable line in Phase 2.
