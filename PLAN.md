# Evidence research agent: build plan

Goal: ask a mental health research question, watch the agent search PubMed and read abstracts
step by step, and get back a cited answer with an honest evidence-strength rating.

The three pieces: **tools** (Python functions it can ask for), **the model** (decides what to
do next), and **the loop** (my code going back and forth until it's done).

---

## Step 0: Setup (10 min)
- venv, `pip install anthropic requests`, set `ANTHROPIC_API_KEY`
- Create `tools.py`, `agent.py`, `evals.py`
- [x] Checkpoint: `python -c "import anthropic"` runs cleanly

## Step 1: Search tool (40 min)
- `search_pubmed(query, max_results, study_types, min_year)`
- esearch gets IDs, then esummary gets title, journal, date, publication types
- Study-type filters (`meta-analysis[pt]`, `randomized controlled trial[pt]`), year filter, rate limiting
- Handle zero results
- Lesson: tools are just functions, and their quality caps the agent's quality
- [x] Checkpoint: prints 5 real meta-analysis titles for "EMDR PTSD"

## Step 2: Reading tool (40 min)
- `fetch_abstracts(pmids)` via efetch (XML)
- Extract title, year, publication types, abstract (keep the section labels), and truncate long ones
- Lesson: every token handed to the model costs money and attention
- [x] Checkpoint: prints two clean, readable abstracts

## Step 3: One model call with tools (30 min)
- Describe both tools as JSON schemas (name, description, inputs)
- One API call with a question plus the tools. Don't execute anything; just inspect the response
- Look for `stop_reason == "tool_use"` and a tool_use block with name, input, id
- Lesson: the model only asks. My code decides whether to act
- [x] Checkpoint: print the tool call the model requested

## Step 4: The loop (45 min)
- Run the requested tool, send back a tool_result with the matching id, call the model again
- Repeat until it stops asking for tools. Keep the full message history
- Max-steps cap, and print each step
- Lesson: this loop is the whole agent
- [x] Checkpoint: one question in, several visible steps, a cited answer out

## Step 5: Behavior (45 min)
- System prompt: strongest evidence first, read before citing, stop when more searching won't
  change the answer, fixed answer format (bottom line / evidence strength / findings with PMIDs /
  caveats), note it's based on abstracts, hand off crisis messages to 988 instead of researching
- Tune the tool descriptions
- [x] Checkpoint: 5 varied questions, noticeably better than in Step 4

## Step 6: Guardrails (45 min)
- Tool errors go back to the model (`is_error`) instead of crashing
- Step cap hit means force a final answer with tools off
- Citation checker in code: every PMID in the answer must have been fetched
- Caching and retries for PubMed
- [x] Checkpoint: break it on purpose (bad search, tiny step cap, crisis message) and it copes

## Step 7: Traces and evals (60 min)
- Log every run to `traces.jsonl` (question, steps, tokens, answer)
- About 15 test questions: settled answers, thin-evidence ones, a couple of safety cases. Run each 3x
- Metrics: citation validity, strength rating vs. expected, tool calls, cost, safety handling
- [x] Checkpoint: a results table plus a list of the failure modes I saw

## Step 8: Write-up (20 min)
- One page: how it works, tradeoffs (abstracts only, no framework, step cap), what failed, what's next
- [ ] Checkpoint: I can explain the whole thing out loud in 3 minutes

---

## Stretch goals
- `fetch_full_text` for free PubMed Central papers (the agent decides when it's worth the extra effort)
- LLM-as-judge grading alongside the rule-based checks
- Rebuild in LangGraph and compare what the framework gives me
- An "evidence behind this approach" feature in Therapize
