import datetime
import json
import re
import time

import anthropic
from dotenv import load_dotenv

from tools import search_pubmed, fetch_abstracts, ALLOWED_STUDY_TYPES, MAX_RESULTS_LIMIT

load_dotenv()

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096
TRACE_FILE = "traces.jsonl"  # one JSON object per run

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SYSTEM_PROMPT = """\
You are a research assistant that answers mental health questions using evidence from PubMed. \
The people asking may be clinicians, students, or people looking into a treatment for themselves \
or someone close to them. They will act on what you say, so an honest account of how strong the \
evidence is matters more than a confident-sounding answer.

# How to research

Look for the strongest evidence first. Start with meta-analyses and systematic reviews. Move to \
randomized controlled trials only if reviews are missing, out of date, or don't address the \
question, and to observational studies only if there are no trials. If a search returns little, \
try broader or alternative terms before concluding the evidence is thin.

Search results show only titles and publication types. Use them to pick the handful of articles \
most likely to answer the question, then read those abstracts with fetch_abstracts. Reading four \
to eight abstracts is usually enough.

Stop searching when another search is unlikely to change your answer: when the reviews you have \
read agree, or when two or three differently worded searches have found nothing better. Finding \
little evidence is a valid result, and you should report it as such.

# What you may claim

Every factual claim about the research must come from an abstract you fetched in this \
conversation, and must carry that article's PMID. Do not cite an article you have only seen in \
search results, and do not add findings, trial names, or numbers from your own memory, even when \
you are confident in them. If something important is missing from what you read, say in the \
caveats that you did not find it; do not fill the gap yourself.

# Answer format

Write the final answer in exactly this structure:

**Bottom line:** One or two sentences that answer the question directly.

**Evidence strength:** One of Strong, Moderate, Limited, or Insufficient, followed by one \
sentence explaining the rating.

**Key findings:**
- One finding per bullet, with the numbers that matter (sample size, effect size), ending with \
its source in the form (PMID: 12345678).

**Caveats:**
- The limitations a reader needs in order to weigh the findings.

*Based on PubMed abstracts only; full texts and clinical guidelines were not reviewed. This is \
not medical advice.*

Use these definitions for the rating:
- Strong: several meta-analyses or large trials agree, with low risk of bias.
- Moderate: reviews or trials support the finding, but with small samples, inconsistent \
results, or quality concerns.
- Limited: a few small or low-quality studies, or evidence only from populations that differ \
from the one asked about.
- Insufficient: little or no relevant research found.

Rate the evidence for the specific question asked. A treatment can have strong evidence that it \
beats no treatment and only limited evidence that it beats an alternative.

# When someone may be in crisis

If the message suggests the person may be thinking about suicide or self-harm, or is otherwise \
in immediate danger, do not search and do not use the answer format. Their safety comes before \
their question. Reply briefly and warmly: acknowledge what they have said, and tell them they \
can call or text 988 (the Suicide & Crisis Lifeline in the US) to talk with someone now, or \
contact local emergency services or a local crisis line if they are elsewhere or in immediate \
danger. Do not summarize research in that reply. You can offer to look into the research later \
if they would like.

This applies to the person's own situation. A question about the research on suicide prevention \
or self-harm treatments, asked without signs of personal crisis, is a normal research question.
"""


SEARCH_TOOL_SCHEMA = {
    "name": "search_pubmed",
    "description": (
        "Search PubMed for articles matching a query. Returns the total number of matches and, "
        "for the most relevant ones, the PMID, title, year, publication types, journal, first "
        "author, and whether an abstract is available. It does not return abstracts. Use this "
        "first to find candidate articles, then use the titles and publication types to choose "
        "which PMIDs are worth reading with fetch_abstracts. Results are sorted by relevance, "
        "and total_found says how many matches exist beyond the ones returned. Several terms in "
        "one query must all match, so long queries return few results; if a search finds little, "
        "retry with fewer or more general terms."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "PubMed search terms: a few key concepts, not a full sentence. "
                    "For example 'EMDR PTSD' rather than 'Is EMDR effective for PTSD?'. "
                    "PubMed syntax such as AND, OR and quoted phrases is allowed."
                ),
            },
            "study_types": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_STUDY_TYPES},
                "description": (
                    "Only return articles of these publication types. An article matches if it "
                    "is any one of them. Omit to search all publication types."
                ),
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_RESULTS_LIMIT,
                "description": "Maximum number of articles to return. Defaults to 10 if omitted.",
            },
            "min_year": {
                "type": "integer",
                "description": (
                    "Earliest publication year to include (inclusive). Defaults to 2015 if omitted. "
                    "Set it lower to find older landmark trials."
                ),
            },
            "max_year": {
                "type": "integer",
                "description": "Latest publication year to include (inclusive). Defaults to the current year if omitted.",
            },
        },
        "required": ["query"],
    },
}

FETCH_TOOL_SCHEMA = {
    "name": "fetch_abstracts",
    "description": (
        "Fetch the abstracts for specific PubMed articles. Returns the PMID, title, year, "
        "publication types and abstract text for each one; long abstracts are truncated. "
        "Only pass PMIDs that were returned by search_pubmed in this conversation; never guess "
        "or recall a PMID from memory. Read an article's abstract with this tool before citing it. "
        "Skip articles whose search result has has_abstract false, since there is nothing to read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pmids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
                "description": "PMIDs to fetch, as strings, exactly as returned by search_pubmed.",
            },
        },
        "required": ["pmids"],
    },
}

TOOLS = [SEARCH_TOOL_SCHEMA, FETCH_TOOL_SCHEMA]

# Sent with the last tool results when the step cap is hit
STEP_CAP_NOTICE = (
    "You have reached the limit on research steps and cannot search or fetch anything more. "
    "Write your final answer now in the required format, using only the abstracts you have "
    "already read. If you could not read enough to answer well, say so and rate the evidence "
    "accordingly."
)

# Matches "PMID: 123" and lists such as "PMIDs: 123, 456"
PMID_PATTERN = re.compile(r"PMIDs?:?\s*(\d+(?:\s*(?:,|;|and)\s*\d+)*)")


# Maps the tool name the model sends to the function that implements it
TOOL_FUNCTIONS = {"search_pubmed": search_pubmed, "fetch_abstracts": fetch_abstracts}


def run_tool(name, tool_input):
    """Run one requested tool. Always returns a dict; failures have an "error" key."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_FUNCTIONS)}"}
    try:
        return func(**tool_input)
    except TypeError as e:
        # The model sent arguments the function doesn't accept
        return {"error": f"Invalid arguments for {name}: {e}"}


def summarize(result):
    """One-line description of a tool result for the step log."""
    if "error" in result:
        return f"error: {result['error']}"
    if "articles" in result:
        return f"{len(result['articles'])} abstracts"
    return f"{len(result.get('results', []))} of {result.get('total_found', 0)} matches"


def call_model(messages, trace, allow_tools=True):
    """One model call. With allow_tools=False the model must answer in text."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        # The schemas are still sent when tools are off, because the history refers to them
        tools=TOOLS,
        tool_choice={"type": "auto" if allow_tools else "none"},
        messages=messages,
    )
    trace["model_calls"] += 1
    trace["input_tokens"] += response.usage.input_tokens
    trace["output_tokens"] += response.usage.output_tokens
    trace["stop_reason"] = response.stop_reason
    return response


def response_text(response):
    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "max_tokens":
        text += " [truncated]"
    elif response.stop_reason == "refusal":
        text = text or "[The model declined to answer this request.]"
    return text


def cited_pmids(text):
    """PMIDs cited in an answer, from forms like (PMID: 123) and (PMIDs: 123, 456)."""
    pmids = set()
    for group in PMID_PATTERN.findall(text):
        pmids.update(re.findall(r"\d+", group))
    return pmids


def verify_citations(answer, fetched_pmids, messages, trace):
    """Make sure every cited PMID was fetched. Gives the model one chance to fix its answer."""
    unverified = cited_pmids(answer) - fetched_pmids
    if not unverified:
        return answer

    trace["citation_revision"] = True
    print(f"[citation check] cited but never fetched: {', '.join(sorted(unverified))}. Asking for a revision.")
    allowed = ", ".join(sorted(fetched_pmids)) or "none"
    messages.append({"role": "user", "content": (
        f"Your answer cites PMIDs whose abstracts were not fetched in this conversation: "
        f"{', '.join(sorted(unverified))}. You cannot search or fetch anything more. Rewrite the "
        f"complete answer now in the same format, removing those citations and any claims that "
        f"depend on them. The only PMIDs you may cite are: {allowed}. The reader will see only "
        f"the rewritten answer and never saw the first one, so do not mention the removed "
        f"citations or this correction."
    )})
    response = call_model(messages, trace, allow_tools=False)
    messages.append({"role": "assistant", "content": response.content})
    answer = response_text(response)

    unverified = cited_pmids(answer) - fetched_pmids
    if unverified:
        print(f"[citation check] still unverified after revision: {', '.join(sorted(unverified))}")
        answer += (
            f"\n\n[Warning: the following cited PMIDs could not be verified against the abstracts "
            f"that were read: {', '.join(sorted(unverified))}.]"
        )
    return answer


def run_agent(question, max_iterations=5):
    """Run the agent on one question. Returns the trace and appends it to traces.jsonl."""
    start = time.time()
    trace = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "model": MODEL,
        "max_iterations": max_iterations,
        "steps": [],
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "stop_reason": None,
        "step_cap_hit": False,
        "citation_revision": False,
    }
    messages = [{"role": "user", "content": question}]
    fetched_pmids = set()

    for step in range(1, max_iterations + 1):
        response = call_model(messages, trace)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = run_tool(block.name, block.input)
            print(f"[step {step}] {block.name}({block.input}) -> {summarize(result)}")
            trace["steps"].append({
                "step": step,
                "tool": block.name,
                "input": block.input,
                "result": summarize(result),
                "is_error": "error" in result,
            })
            # Record what actually came back, not what was requested
            fetched_pmids.update(a["pmid"] for a in result.get("articles", []))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": "error" in result,
            })
        messages.append({"role": "user", "content": tool_results})

    # The loop ran out while the model still wanted tools: force an answer from what it has
    if response.stop_reason == "tool_use":
        print(f"[step cap] {max_iterations} steps used. Forcing a final answer with tools off.")
        trace["step_cap_hit"] = True
        messages[-1]["content"].append({"type": "text", "text": STEP_CAP_NOTICE})
        response = call_model(messages, trace, allow_tools=False)
        messages.append({"role": "assistant", "content": response.content})

    answer = verify_citations(response_text(response), fetched_pmids, messages, trace)

    cited = cited_pmids(answer)
    trace.update({
        "fetched_pmids": sorted(fetched_pmids),
        "cited_pmids": sorted(cited),
        "unverified_pmids": sorted(cited - fetched_pmids),
        "seconds": round(time.time() - start, 1),
        "answer": answer,
    })
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return trace


def agent(question, max_iterations=5):
    return run_agent(question, max_iterations)["answer"]


if __name__ == "__main__":
    print(agent("How effective is EMDR for treating PTSD in adults?"))

