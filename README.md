# Small model, safe actions

**Results page: https://eshwarpotturi.github.io/slm-opti/**

A small open model (Gemma 3 4B) is asked to act on a merchant's account at an online payments platform: refund an order, send an invoice, cancel a subscription. On its own it gets about half the requests right and often takes actions it should not. With five layers of ordinary code around it, the same model gets 89% right on requests it has never seen, and wrong money actions fall from 48 to 3.

Nothing about the model changes: no fine-tuning, and the instruction text stays the same apart from the output wrapper.

## Results

All figures are for Gemma 3 4B, 150 requests per test set.

| Test set | Model alone | Layers 1 to 4 | All 5 layers | Wrong write actions: alone, then all 5 |
|---|---|---|---|---|
| C. Unseen, reworded by another model | 55% | 78% | **89%** | 48, then 3 |
| B. Fresh, 120 of 150 reworded | 50% | 76% | 86% | 57, then 4 |
| A. Original, written alongside the code | 54% | 85% | 97% | 58, then 0 |

**Set C is the number to quote.** The code was finished before set C existed.

Set A is inflated, because the code was developed against it. Set B exposed a splitter that relied on the joining words used in set A; the splitter was rewritten, and set C tests the rewrite. Sets A and B were not rerun with the final code.

### Every layer on the unseen set C, three small models

| Model | Alone | + 1 | + 2 | + 3 | + 4 | Wrong write actions: alone, then layers 1 to 4 |
|---|---|---|---|---|---|---|
| Gemma 3 4B | 55% | 55% | 62% | 65% | 79% | 48, then 3 |
| Llama 3.1 8B | 45% | 59% | 65% | 71% | 85% | 13, then 3 |
| Ministral 3B | 60% | 59% | 71% | 68% | 83% | 25, then 3 |

Each cell is a separate run of 150 requests. Repeated runs of the same setup differ by a few requests: Gemma with layers 1 to 4 scored 78% in one run and 79% in another. Layer 5 has been run on Gemma only.

A "wrong write action" is a refund, cancellation, send or similar that the correct answer does not contain.

## The five layers

| Layer | What the code does | Mistake it prevents |
|---|---|---|
| 1. Format repair and template check | Repairs a malformed reply, checks it against a template built from the tool list, retries once | Correct action written in a form a program cannot read |
| 2. Record grounding | Finds the customer or ID named in the request and shows the model only those records; IDs can only come from them | Acting on the wrong customer's record, invented IDs |
| 3. Split and retrieve | Cuts a message into separate chores; shows the model the few most relevant tools, not all 23 | Forgetting half of a two-part message, picking a lookalike tool |
| 4. Rule check | Checks every proposed action against the account before it runs; rejects with a reason, lets the model correct once, then drops it | Refunding an order already refunded, reminding a paid invoice, inventing an amount |
| 5. Escalation | Asks the small model three times; if the answers disagree, or a rule fired, a large model takes that chore | The remaining hard cases |

On set C, 42 of 150 requests (28%) were escalated to Claude Sonnet 4.5.

### What is generic and what is not

- Layers 1, 3 and 5 know nothing about payments. They are built from whatever tool list is supplied.
- Layer 2 needs a six-line mapping of which ID field points at which kind of record.
- Layer 4 is business rules, about 100 lines for these 23 tools. Any deployment has to write its own; they were written from the tool descriptions.

## Cost

Estimated from list prices, per 150 requests on set C:

| Setup | Cost |
|---|---|
| Small model with layers 1 to 4 | $0.006 |
| All five layers | $0.16, of which $0.15 is the large model's 42 requests |

A large model alone on the same requests has not been measured, so there is no reference line yet.

## Limits

- **No large-model reference.** "Close to a large model" is not shown.
- **The author of the tests also wrote the rules.** Sets B and C use new accounts and wording by a different model, but the kinds of request are the same as in set A.
- **Layer 1 is not true constrained decoding.** It repairs and validates after the fact. Provider-side enforcement was rate-limited during this work.
- **Layers 1 to 3 without layer 4 can reduce safety.** Once Llama 3.1 8B's replies became readable, it acted more often, wrongly too: 13 wrong write actions rose to 35 on set A. Layer 4 brought that to 1.
- **The data is invented.** No real account, customer or transaction is used, and no action is executed; the test checks which tool calls the model proposes.
- **The 23 tools are modelled on a public agent toolkit from a payments provider; the parameters are simplified.**
- **Names are removed in this public copy.** The recorded runs used the platform's name in the one-line prompt and in a tool description; here it reads "an online payments platform". Where a model's reply mentioned the name, it is replaced with "the platform".
- **Exact-match scoring.** A harmless extra lookup counts as a failure.

## Files

| File | Purpose |
|---|---|
| `tools.json` | The 23 actions |
| `tasks.json`, `tasks_fresh.json`, `tasks_third.json` | Test sets A, B and C, each with its answer key |
| `make_tasks.py` | Generates a test set from a seed |
| `paraphrase.py` | Rewords a test set with a different model |
| `stack.py` | The five layers and the runner |
| `score.py` | Parsing and marking |
| `run_eval.py`, `report.py`, `ladder_report.py` | Baseline runner and score tables |
| `build_page.py`, `page_template.html`, `index.html` | The results page, built from `results/` |
| `results/` | Every model reply and its mark |

## Running it

The scripts call models through an API gateway that fronts Amazon Bedrock. Create a file named `.env` (ignored by git) with your gateway token on the first line and a second line `GATEWAY_BASE=<your gateway address>`.

```bash
# model alone, then layers 1 to 4, then all five
python3 stack.py bedrock:google.gemma-3-4b-it --tasks tasks_third.json --layers 0 --tag t
python3 stack.py bedrock:google.gemma-3-4b-it --tasks tasks_third.json --layers 1234 --tag t
python3 stack.py bedrock:google.gemma-3-4b-it --tasks tasks_third.json --layers 12345 --tag t \
    --big bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

Runs are resumable: rerun the same command to continue. Only the Python standard library is needed.

## Questions

**Is this prompt engineering?** No. The instruction text is the same in every run except for the output wrapper. The gain comes from code that limits what the model can see, checks what it proposes, and hands off what it is unsure of.

**Does it exist already?** Each technique is known: schema validation, entity grounding, tool retrieval, guardrails, model cascades. This repository measures them stacked on one small model, layer by layer, on a payments toolkit.

**Why not fine-tune?** Fine-tuning is the obvious next step and has not been tried here.
