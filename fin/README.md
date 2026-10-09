# Small models on financial-report questions

**Task.** Read one page of a company's annual report (text and a table) and work out a figure from it. The questions and answers are from [FinQA](https://github.com/czyssrs/FinQA) (Chen et al., 2021, MIT licence), cut down to its banks, insurers and payment companies: 297 test questions.

**Scoring.** Correct means within 1% of the benchmark's answer. A share may be given as 0.07 or as 7 percent.

## Results on the 297 test questions

| Setup | Correct | Cost per 1,000 questions |
|---|---|---|
| Gemma 3 4B alone | 54% | $0.06 |
| Ministral 3B alone | 63% | $0.05 |
| Both small models cross-checked; a larger model answers where they differ | 78% | $1.16 |
| Gemma 3 4B fine-tuned, alone | 59% | not priced |
| Fine-tuned Gemma and Ministral cross-checked; larger model where they differ | 80% | $1.14 plus running the tuned model |
| Larger model alone (Claude Haiku 4.5) | 80% | $2.01 |

The two small models gave the same answer on 145 of 297 questions, and 86% of those answers were correct. The other 152 went to the larger model, which got 71% of them right.

## What was tried and did not help

On 100 questions from FinQA's development split, with Gemma 3 4B. None was carried to the test.

| Setup | Correct |
|---|---|
| Alone, thinking step by step | 64% |
| Calculator: code redoes the model's arithmetic | 57% |
| Calculator plus a check that every figure is printed in the report | 58% |
| Those two plus four similar solved questions | 64% |
| Similar solved questions only | 53% |
| Five attempts, most common answer wins | 58% |
| Table rewritten one labelled row per line | 57% |

Most wrong answers come from choosing the wrong figure or misreading the question, not from arithmetic.

## Fine-tuning

`finetune_fin.ipynb` trained Gemma 3 4B for 36 minutes on a free Colab GPU, on 1,026 solved questions from the training split, to write the formula directly. Code then computes the formula and checks its figures against the report.

- Alone: 174 of 297 correct (59%), up from 54% untuned.
- Cross-checked with Ministral 3B: the two agreed on 144 questions and 133 of those answers (92%) were correct. With the larger model on the other 153, the total is 239 of 297 (80%).

The fine-tuned model writes shares as decimals and the other model writes them as percents, so for this pair two answers count as agreeing when they match after that conversion. The escalated answers are taken from the larger model's recorded run on the same questions.

## Limits

- About half the gain comes from the larger model.
- The benchmark supplies the right page. Finding it in a full report is not tested.
- FinQA's own answers contain some errors.
- Single runs.

## Files

- `prep.py` cuts the benchmark down. `data/` holds the result.
- `run.py` runs one model, alone or with the layers that were tried.
- `crosscheck.py` combines two small models and escalates disagreements.
- `build_page.py` writes `index.html`.
- `results/` holds every recorded answer.
