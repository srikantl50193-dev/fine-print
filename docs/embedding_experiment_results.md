# Embedding models in Fine Print: tuning-set results

Run by `scripts/compare_embedding_models.py`: section chunks, top 5 passages, gemma4:e4b with the `numbered` prompt. Tuning set only: 31 answerable and 15 out-of-scope questions. Fine Print's held-out set is spent and wasn't used.

**Rows marked "re-tuned" chose their threshold on these same questions and were scored on them, so they are optimistic.** Rows at 0.6 use Fine Print's existing value.

| Embedding model | min_similarity | Retrieval recall@5 | MRR | Correct citation | Correct-citation wins / losses vs bge-small at its matching threshold | Out-of-scope refused | Answerable wrongly refused | Answer p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small | 0.6 (current) | 20/31 | 0.618 | 20/31 | - | 15/15 (8 threshold, 7 model) | 1 (0 threshold, 1 model) | 8.5 s |
| bge-base | 0.6 (current) | 23/31 | 0.596 | 22/31 | 2 / 0 (p 0.50) | 15/15 (9 threshold, 6 model) | 1 (0 threshold, 1 model) | 8.1 s |
| bge-small-fine-tuned | 0.6 (current) | 22/31 | 0.603 | 16/31 | 0 / 4 (p 0.12) | 15/15 (15 threshold, 0 model) | 5 (4 threshold, 1 model) | 6.8 s |
| bge-small | 0.738 (re-tuned) | 20/31 | 0.618 | 19/31 | - | 15/15 (15 threshold, 0 model) | 3 (2 threshold, 1 model) | 5.7 s |
| bge-base | 0.655 (re-tuned) | 23/31 | 0.596 | 20/31 | 1 / 0 (p 1.00) | 15/15 (15 threshold, 0 model) | 2 (1 threshold, 1 model) | 6.6 s |
| bge-small-fine-tuned | 0.571 (re-tuned) | 22/31 | 0.603 | 19/31 | 2 / 2 (p 1.00) | 15/15 (15 threshold, 0 model) | 3 (2 threshold, 1 model) | 7.9 s |

- **Retrieval recall@5** is Fine Print's keyword measure on the section index: the right document and every keyword in a top-5 chunk. It doesn't depend on the threshold.
- **Correct citation** means the answer cites the section of a passage that passes the keyword test. Fine Print's own caveat applies: it doesn't check the answer is right.
- **Wrongly refused** answerable questions are split by cause: below `min_similarity`, so the model never saw them, or declined by the model.
- **Out-of-scope refused** is split the same way. Where the model declines them itself, a higher threshold adds nothing but wrong refusals.
- One question is 3.2 points of 31 answerable questions; gaps of one or two are noise. Answer latency includes the real model, measured with no other apps open.
