# Mental health evidence research agent: write-up

## Summary

I built an agent that answers mental health research questions from PubMed. You ask something like "How effective is EMDR for PTSD in adults?". The agent searches PubMed, reads abstracts, and returns a cited answer with an honest rating of how strong the evidence is.

It is written by hand in Python with the `anthropic` SDK and `requests`, with no agent framework. On an eval of 15 questions run 3 times each, 42 of 45 runs passed. Every cited paper had been read by the agent, and every crisis message was handled safely. A run costs about $0.056 and takes about 14 seconds.

## How it works

An agent has three parts:

- **Tools:** Python functions the model can ask for.
- **The model:** decides what to do next.
- **The loop:** my code, which goes back and forth between the two until the model is done.

One run looks like this:

1. I send the model the question, a system prompt, and descriptions of two tools.
2. The model replies with a request: "call `search_pubmed` with these arguments". It does not run anything; it only asks.
3. My code runs the function and sends the result back, tagged with the id of the request.
4. The model reads the results and asks for the next tool, usually `fetch_abstracts` on the papers that look most relevant.
5. This repeats until the model replies with text instead of a tool request. That text is the answer.
6. My code checks the citations in the answer before returning it.

A typical run is three model calls: two searches in parallel, one fetch of 5 to 8 abstracts, then the answer.

The model is stateless. It remembers nothing between calls, so I resend the whole conversation every time. Its only memory is the `messages` list that my code owns.

## The pieces

### Tools (`tools.py`)

- **`search_pubmed`:** takes a query, optional study types, a year range and a result limit. It returns each paper's PMID, title, year, publication types, journal and first author, plus `total_found`, the number of matches in all of PubMed.
- **`fetch_abstracts`:** takes a list of PMIDs and returns each abstract with its section labels, truncated at 2,500 characters.

Both go through one function, `_get`, which handles the things PubMed requires:

- **Rate limiting:** at least 0.34 seconds between requests, because PubMed allows 3 per second without an API key.
- **Retries:** up to 3 attempts with back-off on rate-limit errors, server errors and network failures.
- **Caching:** an identical request within one run is answered from memory.

The tools never raise on failure. They return `{"error": "..."}` with a message the model can act on, such as "No matches. Try broader terms."

### Tool schemas (`agent.py`)

Each tool is described to the model with a name, a description and a JSON Schema for its inputs. The model reads only this, never my code, so the descriptions work as prompts. Study types are an `enum`, so the model can only choose from the six valid values.

### The loop (`agent.py`)

`run_agent` is about 60 lines. It calls the model, appends the reply to the history, runs any requested tools, appends the results, and repeats. A dictionary maps each tool name to its function, so adding a tool is one schema and one dictionary entry.

### System prompt

The prompt tells the model:

- to look for the strongest evidence first: reviews, then trials, then observational studies
- to claim only what is in an abstract it fetched, with the PMID
- when to stop searching
- to use a fixed answer format: bottom line, evidence strength, key findings, caveats
- to rate evidence with one of four defined words: Strong, Moderate, Limited, Insufficient
- to respond to a person in crisis with support and the 988 lifeline, and no research

### Guardrails in code

- **Step cap:** the loop stops after 5 rounds of tool use. A final call with tools switched off then forces an answer from what the agent has already read.
- **Citation checker:** the loop records every PMID that `fetch_abstracts` returned. If the answer cites a PMID that is not in that set, the model gets one chance to rewrite. If the rewrite still fails, a visible warning is added.
- **Tool errors:** an unknown tool name or bad arguments go back to the model as an error it can correct. Bugs in my own code are not caught, so they crash loudly.
- **Input limits:** the result limit is clamped in code, because the model usually respects the schema but is not guaranteed to.

### Traces and evals (`evals.py`)

Every run appends one line to `traces.jsonl` with the question, each tool call, token counts, fetched and cited PMIDs, and the answer. The eval runs 15 questions 3 times each and grades every run with yes/no checks in code.

## Design decisions and tradeoffs

| Decision | Why | What it costs |
|---|---|---|
| No framework | The loop is the whole agent, and writing it by hand shows exactly what a framework would hide. | I own the retries, state and tracing myself. |
| Abstracts only | Abstracts are free, short and available for almost every paper. | Effect sizes, methods and limitations in the full text are invisible. Every answer says so. |
| Two tools | A small tool set is easy for the model to use correctly. | No full text, no guidelines, no other databases. |
| Step cap of 5 | It bounds cost and stops endless searching. | A hard question may be answered from less reading. |
| Citation check in code | A prompt is a request. Code is a guarantee. | It checks that a paper was read, not that the claim matches it. |
| Fixed answer format | The rating and citations can be parsed and graded automatically. | Answers are less conversational. |
| In-memory cache | It makes repeated eval runs faster and kinder to PubMed. | It never expires, so it is only safe for a single run. |
| Tools return errors as data | The model can recover by rewording a search. | I have to set the error flag myself from the result. |

## Results

**Before and after the system prompt** (5 questions, one run each):

| | Before | After |
|---|---|---|
| Citations | Author and year, no PMIDs | Every finding has a PMID |
| Evidence rating | None | On every research answer |
| Answer format | Different each time | The same four sections |
| Claims from memory | One answer cited a trial from "background knowledge" | The agent searched for the trial, read it and cited it |
| Crisis message | 988 plus an uncited research summary | 988 and support only |

**Eval** (15 questions, 3 runs each, 45 runs):

| Check | Result |
|---|---|
| Runs passed | 42/45 |
| Every cited PMID was fetched | 39/39 |
| Answer has all four sections | 39/39 |
| Rating within the expected range | 36/39 |
| Crisis handled with no research and 988 | 6/6 |
| Research question about suicide answered normally | 3/3 |
| Step cap hit | 0/45 |
| Cost | $2.50 total, about $0.056 per run |
| Time | about 14 seconds per run |

## What failed

1. **One question failed all three runs.** For weighted blankets and anxiety, the agent said Moderate and I had expected Limited. It found two recent meta-analyses, which fits Moderate, but the patients were mostly psychiatric or surgical, which fits Limited. The disagreement may be in my expected answer, not the agent.
2. **Abstracts get cut off.** 12 of 45 answers mentioned a truncated abstract. The 2,500-character limit sometimes removes the results or conclusion.
3. **One answer cited a paper it never read.** The citation checker caught it and the rewrite fixed it. This showed that the prompt alone does not guarantee the rule.
4. **Ratings cluster on Moderate.** Only 5 of 18 runs on well-established treatments said Strong. The agent may be too cautious, or abstracts may not carry enough to justify Strong.
5. **Search effort varies.** Most questions took 3 tool calls. A few took 7 and cost twice as much. Those are the ones that will hit the step cap.

Bugs I hit while building:

- **A missing import looked like a PubMed outage.** A broad `except` caught a `NameError` in my own code and reported it to the model as a tool failure. The model then told me it could not reach PubMed. I narrowed the `except` so my own bugs crash.
- **An unreachable error message.** The retry logic had a line that could never run, so a persistent rate limit reported the wrong message.
- **The citation rewrite flagged itself.** The model wrote "I removed PMID X", and the checker then counted X as cited. The rewrite instruction now says not to mention the correction.

## Limitations

- **Faithfulness is not measured.** The eval confirms a cited paper was read. It does not confirm the paper says what the answer claims.
- **Expected ratings are my own.** No clinician verified them.
- **The eval is lenient and small.** Each case accepts two of the four ratings, and three runs per question cannot catch rare failures.
- **Crisis detection is prompt-only.** There is no check in code behind it.
- **It is not medical advice,** and every answer says so.

## What's next

- **An LLM judge for faithfulness:** a second model checks each claim against the abstract it cites.
- **Full text** for open-access papers in PubMed Central, with the agent deciding when it is worth the extra tokens.
- **A longer or smarter abstract limit,** so that conclusions are never cut.
- **Clinician-reviewed expected answers** and more eval cases.
- **A rebuild in LangGraph** to compare what a framework gives me against what I wrote.
