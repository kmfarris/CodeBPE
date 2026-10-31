# How to read the numbers

Every experiment in this repo uses a recorded dataset, model size, and gate script. The files next to each run (`results.json`, `results.md`) are the source for tables in the README and project summary.

## What the runs established

On WikiText-2-scale **bucket+slot** codes, a one-word edit stays in a small region of the code stream, and decoding returns the original text (`experiments/working_lm/results.json`).

With two copies of the same small language model (six layers, 384-wide hidden states), training on CodeBPE-over-buckets uses fewer bits per character on the **training** split than training on ordinary text BPE. That comparison holds the model body fixed so the alphabet is what differs.

Direction 3, a coarser-merge ladder on a ~100k-word slice, missed its coarseness gates. Those logs stay in the tree so later merge work has a floor, not only the passing WikiText-2 run.

## What still needs work

- **Web-scale and other languages** are not measured here.
- **Cost vs industrial tokenizers** (SentencePiece and similar) at large token counts is not measured here.
- **Phrase-sized pieces:** after frequency-filtered merges, a typical merged unit is still about 1.4 codes long—closer to a word than a long phrase.
- **Validation packing** is looser than train packing (more unknown words to spell out on held-out text). Always quote both columns from `experiments/working_lm/results.json`.
- **Vocabulary size** changes by experiment. Early directions used smaller vocabs; `working_lm` uses the full WikiText-2 word list. Mixing tables without that note misleads.
- **Model checkpoints** (`.pt`) are not in git. Rebuild with `python -m experiments.working_lm.run`.

A write-up that compared SentencePiece or unigram on the same split, with validation packing next to train packing in one script output, would close the most important remaining baseline gap.
