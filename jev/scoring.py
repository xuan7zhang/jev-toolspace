"""Benchmark-agnostic Jev tool judgments.  Every call scores all candidate tools of one request with one independent
noul question per tool (never a choice question), so a tool's probability does not compete with the others.
Tool order in the state is the caller's canonical order and is recorded with the prompts."""
import statistics as st
PROMPTS = {
 "request": ("For the user request `request`, would an agent need to call the tool {tool} (described in `tools`) to answer it "
             "correctly? Count a tool as needed if its output is required at any step, including intermediate steps whose "
             "result feeds another tool. Judge only from the request and the tool descriptions."),
 "trace": ("An assistant answered the request `request` after seeing the outputs in `tool_outputs` and gave the final answer "
           "`answer`. Does that answer rely on the output of the tool {tool}? Say yes if the answer uses information that "
           "appears in that tool's output, directly or through another tool's output that was computed from it. "
           "Do not judge whether the answer is correct and do not answer the request yourself."),
  "conversation": ("An assistant handled the multi-turn user conversation in `conversation` by calling tools; `conversation` lists the user "
           "turns, the assistant's calls, the tool outputs, and the assistant's replies. Did the assistant rely on the tool {tool} to carry "
           "out the user's requests? Say yes if a call to {tool} performed an action the user asked for, or if its output was used by a later "
           "call or by the assistant's reply. Do not judge whether the assistant succeeded, and do not carry out the requests yourself."),
}
def qid(tool): return "t_" + "".join(c if c.isalnum() else "_" for c in tool)

def score_request(client, request, tools, tag=None):
    """tools: list of (name, doc) in canonical order -> ({name: p}, call record)"""
    state = {"request": request, "tools": [{"name": n, "description": d} for n, d in tools]}
    qs = {qid(n): {"type": "noul", "instructions": PROMPTS["request"].format(tool=n)} for n, _ in tools}
    r = client.ask(state, qs, tag=tag); return {n: r["answers"].get(qid(n)) for n, _ in tools}, r

def score_trace(client, request, outputs, answer, tools, tag=None):
    """outputs: list of (name, output text) actually shown to the executor; only tools with an output are judged"""
    state = {"request": request, "tool_outputs": [{"tool": n, "output": o} for n, o in outputs], "answer": answer}
    shown = [n for n, _ in outputs]
    qs = {qid(n): {"type": "noul", "instructions": PROMPTS["trace"].format(tool=n)} for n in shown}
    r = client.ask(state, qs, tag=tag); return {n: r["answers"].get(qid(n)) for n in shown}, r

def top_k(scores, k, order):
    """highest scores first; ties broken by canonical order; tools without a score are never selected"""
    have = [t for t in order if scores.get(t) is not None]
    return sorted(have, key=lambda t: (-scores[t], order.index(t)))[:k]

def aggregate(per_request, order):
    """mean over the requests that scored the tool (missing = no evidence, not zero)"""
    out = {}
    for t in order:
        v = [s[t] for s in per_request if s.get(t) is not None]
        if v: out[t] = st.mean(v)
    return out

def score_conversation(client, conversation, tools, tag=None):
    """conversation: list of {role, content}; tools: names that were called (only these are judged)"""
    qs = {qid(n): {"type": "noul", "instructions": PROMPTS["conversation"].format(tool=n)} for n in tools}
    if not qs: return {}, None
    r = client.ask({"conversation": conversation}, qs, tag=tag); return {n: r["answers"].get(qid(n)) for n in tools}, r
