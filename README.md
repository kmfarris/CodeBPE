# customLLM: a new way to break text into pieces for AI

## What it is

customLLM is a research project that tests a different way of preparing text before an AI language model reads it.

## The problem

AI language models, like the ones behind chat assistants, can't read words directly. Text is first chopped into small pieces called **tokens**, and the model learns to predict one piece after another.

The standard method splits text wherever letter combinations are common. Those pieces don't line up with whole words, and changing one word can reshuffle pieces elsewhere in the sentence. That makes it hard to see what changed in an edited document, to swap one name for another, or to treat a word as the same thing every time it appears.

## How it works

1. **One code per word.** Every known word gets its own fixed code, a bit like a library call number. No two words share a code.
2. **Spell out the unfamiliar.** A word the system hasn't seen before is spelled out letter by letter, so the original text can always be rebuilt exactly.
3. **Join common neighbors.** Codes that often appear side by side are combined into larger pieces. Standard tokenizers do the same thing with letters; here it's done with whole-word codes.
4. **Compare fairly.** Two identical small AI models are trained on the same text. One uses standard tokens and the other uses these codes, so the only difference between them is how the text was broken up.

## Results so far

Tested on WikiText-2, a collection of about two million words from Wikipedia articles:

- **Nothing lost.** Text converted to codes and back came out identical every time.
- **Edits stay local.** Changing a single word only affected the codes immediately around it.
- **Slightly better learning.** The model trained on codes predicted the text a little better. It needed 1.76 bits of information per letter, compared with 1.88 for standard tokens, about 6% less.
- **Slightly tighter packing.** The same text took a little less space as codes than as standard tokens.

Earlier stages also showed that the codes can pinpoint exactly which word changed between two versions of a sentence, and that a person's or city's name can be swapped out in code form and still read correctly. A full stage-by-stage account is in [`project_summary.md`](project_summary.md).

## Still to do

- **Unfamiliar text.** On text the system hasn't seen, more words have to be spelled out, so the packing advantage shrinks.
- **Bigger pieces.** The joined pieces are still roughly one word long. An attempt to build phrase-sized pieces didn't reach its target, and those results are kept in the repository.
- **Scale and comparison.** Testing so far uses one dataset and small models. The method hasn't yet been compared against other modern tokenizers such as SentencePiece, or tried on much larger text collections.

More detail on how to read the numbers: [`docs/SCOPE_AND_LIMITS.md`](docs/SCOPE_AND_LIMITS.md).

---

## For developers

Run the tests:

```bash
pip install -e '.[dev]'
python -m pytest tests/
```

Reproduce the main experiment. It needs a GPU, and the trained models aren't stored in the repository, so this rebuilds them:

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.working_lm.run
PYTHONPATH=src:. python -u -m experiments.working_lm.run --rescore-locality
```

Results: [`experiments/working_lm/results.md`](experiments/working_lm/results.md) · [`results.json`](experiments/working_lm/results.json)

Earlier experiments:

| Stage | Command |
|-------|---------|
| Joined codes + smaller model | `PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir6_bucket_codebpe.run_bakeoff` |
| One code per word | `PYTHONPATH=src:. python -u -m experiments.dir5_bucket_slot.run_bench` |
| Spelling out unfamiliar words | `PYTHONPATH=src:. python -u -m experiments.dir4_oov_decode.run_m1` |
| Phrase-sized pieces (missed target) | `PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir3_scale_ladder.run_ladder --budget 30` |

The earliest stages (A–C) run on an ordinary CPU. Each folder under [`experiments/`](experiments/) has its own `results.md`.
