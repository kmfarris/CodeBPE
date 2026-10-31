# CodeBPE

A tokenizer research project: every word gets a fixed, collision-free code, and byte-pair encoding (BPE) then merges those codes instead of raw characters. A matched language-model comparison against standard text BPE measures how this affects learning.

## Purpose

Language models don't read words. Text is first split into **tokens**, and the model learns to predict the next token. The standard method, byte-pair encoding, starts from characters and repeatedly merges the most frequent adjacent pair. The resulting tokens are statistical fragments: they rarely match word boundaries, the same word can be split differently depending on its surroundings, and a one-word edit can change the tokenization of neighboring text.

That makes token streams hard to work with directly. It's difficult to tell which tokens a given word produced, to diff two versions of a document at the token level, or to substitute one name for another without re-tokenizing the whole passage.

CodeBPE tests whether the tokenizer's alphabet can be made stable and word-aligned without giving up modeling quality. Words become fixed codes first, and BPE merges operate on those codes.

## How it works

1. **Positioned Residual Code Fingerprints (PRCF).** Each vocabulary word is assigned a code made of a *bucket* (a coarse group) and a *slot* (its position within that group). The assignment is injective, meaning no two words share a code, so every code decodes to exactly one word.
2. **Escape spell-out.** Words outside the vocabulary are emitted as an escape marker followed by their characters. Every input can be encoded, and decoding reproduces the original text exactly.
3. **CodeBPE merges.** Standard BPE is trained over the code stream instead of the character stream. Frequently adjacent codes are merged into single tokens, so merges grow from whole-word units.
4. **Matched bake-off.** Two language models with identical architecture (6 layers, 384-dimensional hidden states, the same number of non-embedding parameters) are trained for the same number of steps on the same WikiText-2 data. One uses standard text BPE and the other uses CodeBPE. Because the model body is held fixed, any difference in results comes from the tokenizer.

Quality is scored in **bits per character (BPC)** on held-out validation text. BPC normalizes by characters instead of tokens, so tokenizers with different vocabularies can be compared directly. Lower is better.

## Results

Measured on WikiText-2 (about 2 million training words from Wikipedia):

- **Lossless round trip.** Encoding and then decoding reproduced the original text exactly, including out-of-vocabulary words (chrF 1.00, exact out-of-vocabulary recovery 1.00).
- **Edit locality.** In 300 one-word edit tests, comparing the code streams before and after the edit pinpointed the edited word exactly every time (1.00 exact, against a target of 0.80 within ±2 positions).
- **Better modeling at matched model size.** The CodeBPE model reached 1.76 BPC on validation text, compared with 1.88 for text BPE, which is 6% lower. Validation perplexity was 14.6 vs 17.4.
- **Tighter packing on training text.** CodeBPE used 0.227 tokens per character on the training split, against 0.240 for text BPE.

Earlier stages established that bucket+slot codes localize which word changed between two versions of a sentence, and that entity names can be substituted directly in code form. The stage-by-stage record is in [`project_summary.md`](project_summary.md).

## Areas still needing work

- **Held-out packing.** On validation text, more words fall outside the vocabulary and have to be spelled out, so CodeBPE packs looser than text BPE there (0.268 vs 0.242 tokens per character). The vocabulary is built from the training set, and coverage of unseen text limits how compact the encoding can be.
- **Vocabulary and embedding cost.** The CodeBPE vocabulary has 59,117 entries, against 10,000 for text BPE, so the embedding tables are larger even though the model bodies are identical (33.4M vs 14.6M total parameters).
- **Phrase-level merges.** Merged units still average about 1.4 codes, so they are close to word-sized. A scale-ladder experiment aimed at coarser, phrase-length units missed its targets, and its logs are kept in `experiments/dir3_scale_ladder/`.
- **Scale and baselines.** All results come from one English dataset and small models. There's no comparison yet against SentencePiece or unigram tokenizers, and no measurement at web scale or in other languages.

How to read and compare the numbers: [`docs/SCOPE_AND_LIMITS.md`](docs/SCOPE_AND_LIMITS.md)

---

## For developers

The Python package is `prcf`.

```bash
pip install -e '.[dev]'
python -m pytest tests/
```

Reproduce the main bake-off. It needs a GPU, and model checkpoints aren't stored in the repository:

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.working_lm.run
PYTHONPATH=src:. python -u -m experiments.working_lm.run --rescore-locality
```

Results: [`experiments/working_lm/results.md`](experiments/working_lm/results.md) · [`results.json`](experiments/working_lm/results.json)

Earlier experiments:

| Stage | Command |
|-------|---------|
| CodeBPE over buckets, smaller model | `PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir6_bucket_codebpe.run_bakeoff` |
| Bucket+slot codes | `PYTHONPATH=src:. python -u -m experiments.dir5_bucket_slot.run_bench` |
| Escape spell-out decoding | `PYTHONPATH=src:. python -u -m experiments.dir4_oov_decode.run_m1` |
| Phrase-level merge ladder (missed targets) | `PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir3_scale_ladder.run_ladder --budget 30` |

The earliest stages (A–C) run on CPU. Each folder under [`experiments/`](experiments/) has its own `results.md`.
