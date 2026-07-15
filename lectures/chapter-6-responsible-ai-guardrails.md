# Chapter 6: Responsible AI & Agentic Guardrails

## Overview

Models need to be constrained. A model that answers everything, including what it should refuse or doesn't actually know, isn't being helpful. This chapter adds four checks to the graphs and chains already built: reading confidence as a real signal, refusing by policy, narrating action, and boxing in what a tool can do. Every piece uses existing LangChain and LangGraph capabilities, and none of it is specific to any one system.

## Route on the Score You Already Have

Chapter 4's retrieval already returns a similarity score for the best match. Add a `best_score: float` field to state alongside `retrieved_chunks` from Chapter 5, set by the same retrieve node, and route on it directly as a conditional edge instead of an if-statement buried inside one function.

```python
def route_by_confidence(state: OnboardingState) -> str:
    if state["best_score"] >= CONFIDENCE_THRESHOLD:
        return "generate"
    return "check_scope"

graph.add_conditional_edges("retrieve", route_by_confidence, {"generate": "generate", "check_scope": "check_scope"})
```

Nothing new had to be computed to make this decision. The score was already sitting in state.

Whatever's consuming the API doesn't need the raw number, just whether to trust the answer. Wherever the response actually gets built, the same comparison becomes a bool:

```python
class AskResponse(BaseModel):
    answer: str
    citation: list[str] | None = None
    confident: bool
```

`confident = best_score >= CONFIDENCE_THRESHOLD` is computed once at the point a response is returned.

## Checking Scope Before Refusing

A low score alone doesn't say whether a question is off-topic or something the assistant should actively decline. Ask a narrower question: does this belong to the domain the assistant was built for at all?

```python
class ScopeResult(BaseModel):
    """Whether a question belongs to the assistant's declared domain."""
    in_scope: bool
    reason: str

async def check_scope(state: OnboardingState) -> dict:
    result = await scope_llm.ainvoke(state["question"])
    if result.in_scope:
        return {"final_answer": AskResponse(answer="I don't have a confident answer to that.", confident=False)}
    return {"final_answer": AskResponse(answer=f"That's outside what I can help with: {result.reason}", confident=True)}
```

This node only runs once the confidence edge has already decided retrieval came up empty. A high-confidence match doesn't reach it.

## Narrating What Changed

Say what happened in the same breath as doing it. A tool call's result belongs in the turn that reports back.

```python
async def execute_tool(state: OnboardingState) -> dict:
    result = await dispatch_tool(state)
    return {"tool_result": result, "narration": f"Done: {result['summary']}"}
```

## Two Separate Questions Before a Tool Runs

Every tool call answers two questions before it touches anything. Who is this, and are they allowed to do this. Authentication answers the first. Authorization answers the second. Neither one gets answered by reading the chat.

```python
async def execute_tool(state: OnboardingState) -> dict:
    tool_call = state["last_response"].tool_calls[0]
    tool = TOOLS[tool_call["name"]]
    result = await tool.ainvoke({**tool_call["args"], "user_id": state["user_id"]})
    return {"tool_result": result}
```

`state["user_id"]` was set once, at the start of the request, from the authenticated session. `tool_call["args"]` is just the model's own reading of the conversation, and it isn't allowed to answer either question.

## Boxing In What a Tool Can Actually Do

The API a tool calls is often capable of far more than the tool should be trusted with. Draw the line by choosing which tools exist and telling the model plainly what's off the table, rather than building a permission check to catch it after the fact.

```python
SYSTEM_PROMPT = """You manage a new hire's tasks and plan.
Add, edit, or complete tasks. Revise plan text.
Time off, pay, and compliance items are not yours to touch.
Flag those for a person instead."""
```

There's no `update_salary` tool in the set the model was given. The instruction above is a second layer, not the only one.

## Gotchas

- A rule that only lives in a system prompt is a strong suggestion. A determined user can talk a model out of it. Back every prompt-level rule with a code-level check somewhere it actually matters: a threshold, a fixed tool set, an injected id.
- A model refusing in ten test conversations doesn't mean it refuses in the eleventh. Prose-only guardrails are probabilistic, no matter how many times they've worked so far.
- Reporting an action after it's already written means the user finds out, not that they got a say. Fine for something reversible like a task edit. Not a template for anything bigger without a confirmation step.
- Checking scope only when confidence is low has a hole: a disallowed question that happens to resemble something in the store slips through with high confidence and never reaches the check. Run it on every question instead if that risk is real for your data.

## Quick Reference

| Guardrail | Where it lives |
|---|---|
| Confidence field | `AskResponse.confident` |
| Confidence-first routing | conditional edge off the retrieve node |
| Scope check | LLM call, reached only on low confidence |
| Narration | same dict the tool node already returns |
| Identity | `state["user_id"]`, not `tool_call["args"]` |
| Authority cap | tool set plus system prompt, not a permissions table |
