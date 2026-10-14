# Mental health evidence research agent

An agent that answers mental health research questions from PubMed. Ask something like "How effective is EMDR for PTSD in adults?" and it searches PubMed, reads abstracts, and returns a cited answer with a rating of how strong the evidence is.

The agent loop is written by hand in Python with the `anthropic` SDK and `requests`. No agent framework.

## Example output

A shortened answer from a real eval run:

> **Bottom line:** EMDR works for adults with PTSD. It reduces symptoms more than waitlist or no treatment, and the reviews I read found it roughly as effective as other trauma-focused therapies such as trauma-focused CBT. Many of the underlying trials are of limited quality.
>
> **Evidence strength:** Moderate. Several meta-analyses agree that EMDR beats no treatment and is comparable to other active therapies. However, the authors rate much of the evidence as low to moderate quality, with high heterogeneity and few low-risk-of-bias trials.
>
> **Key findings:**
> - **Versus waitlist or no treatment:** In a network meta-analysis of 90 trials (6,560 people), EMDR had the largest estimated reduction in PTSD symptoms after treatment compared with waitlist (SMD −2.07). The authors rated the evidence as moderate-to-low quality (PMID: 32063234).
> - **Head-to-head with other therapies, individual participant data meta-analysis:** This analysis covered 8 RCTs (346 patients). It found no significant difference between EMDR and other psychological treatments for symptom severity, response, remission, or dropout (PMID: 38173121).
> - *...7 more findings*
>
> **Caveats:** Effect sizes against waitlist are inflated by weak comparators... Adverse events and dropout are poorly reported in most trials...
>
> *Based on PubMed abstracts only; full texts and clinical guidelines were not reviewed. This is not medical advice.*

## How it works

1. The model gets the question, a system prompt and two tools: `search_pubmed` and `fetch_abstracts`.
2. It asks for a tool. The code runs it and sends back the result.
3. This repeats until the model writes an answer, usually after about three model calls.
4. The code checks that every cited PMID was actually fetched. If one wasn't, the model rewrites once, and a warning is added if the rewrite still fails.

Guardrails:
- **Step cap:** after 5 rounds of tool use, a final call with tools switched off forces an answer.
- **PubMed limits:** requests are rate limited to PubMed's 3 per second, retried with back-off and cached.
- **Errors as data:** tool errors go back to the model so it can correct itself.
- **Crisis handling:** a message from someone in crisis gets support and the 988 lifeline, not a literature review.

## Files

| File | What it does |
|---|---|
| `tools.py` | PubMed E-utilities: search, fetch abstracts, rate limiting, retries, cache |
| `agent.py` | System prompt, tool schemas, the loop, citation checker, tracing |
| `evals.py` | 15 test questions, each run 3 times, graded with checks in code |
| `WRITEUP.md` | Design decisions, results, failure modes and limitations |

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install anthropic requests python-dotenv
```

Create a `.env` file containing:

```
ANTHROPIC_API_KEY=your-key-here
```

Then run it:

```bash
python tools.py                 # try the PubMed tools on their own (no API key needed)
python agent.py                 # run the agent on an example question
python evals.py                 # run the full eval: 15 questions x 3 runs (about $2.50)
python evals.py --runs 1        # run each question once
python evals.py --report        # print the results table from the last run
```

To ask your own question:

```python
from agent import agent
print(agent("Does exercise reduce symptoms of depression?"))
```

Every run is logged to `traces.jsonl`.

## Results

15 questions, 3 runs each (45 runs):

| Check | Result |
|---|---|
| Runs passed | 42/45 |
| Every cited PMID was fetched | 39/39 |
| Evidence rating within the expected range | 36/39 |
| Crisis messages handled safely | 6/6 |
| Cost | about $0.056 per run |
| Time | about 14 seconds per run |

See [WRITEUP.md](WRITEUP.md) for the failure modes and what's next.

## Limitations

- **Abstracts only:** the full text of papers is never read.
- **Faithfulness not measured:** the code checks that a cited paper was read, not that it supports the claim.
- **Unverified expected ratings:** no clinician reviewed the ratings the eval expects.
- **Not medical advice.**
