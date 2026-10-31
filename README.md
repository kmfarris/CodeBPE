# customLLM

Language models never see a paragraph as letters. First the text is cut into **tokens**—short pieces the model learns to predict, one after another. Almost every system in use today cuts those pieces by **spelling**: frequent letter patterns become frequent tokens. That works, but it has a cost. Change one word in a sentence and the token sequence can rearrange far from the edit. Names that look similar can share codes they should not. The model’s alphabet is optimized for letter frequency, not for “this word is this word.”

**customLLM** is a research project on a different first cut: **Positioned Residual Code Fingerprints (PRCF)**. Each known word gets a stable code. Unknown words are spelled out through an escape path so nothing is lost. Nearby codes can merge into slightly larger units (CodeBPE), the same idea as classic BPE, but over word codes instead of letters. A small language model then trains on that alphabet and we compare it, fairly, to the same model trained on ordinary text BPE.

The intended outcome is a tokenizer you can measure: edits stay near the change, text round-trips without collisions, and a matched model can still learn language from the new pieces.

## What works

On the full [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) English corpus (about two million training words), with two copies of the same six-layer, 384-wide language model:

- **Round-trip:** encoded text decodes back exactly (invertibility 1.00).
- **Local edits:** changing a single word moves only nearby codes (±2 in the codec’s index space, score 1.00).
- **Density:** the code stream is slightly tighter than text BPE on training data (0.227 vs 0.240 symbols per character).
- **Learning:** bits per character on the training split is **1.76** for the code alphabet vs **1.88** for text BPE.

Earlier stages of the same pipeline already showed that a fingerprint can name *which* word changed, that name and city codes can be transplanted, and that out-of-vocabulary words can be spelled out without smashing two similar in-vocab words onto one id. Numbers and scripts live under [`experiments/`](experiments/); the working-scale table is [`experiments/working_lm/results.json`](experiments/working_lm/results.json). A narrative of every stage is in [`project_summary.md`](project_summary.md).

```bash
pip install -e '.[dev]'
python -m pytest tests/
```

## What still needs work

- **Held-out packing.** On validation text the code stream is looser than on train (more unknown words to spell out). Report train and validation together; see [`docs/SCOPE_AND_LIMITS.md`](docs/SCOPE_AND_LIMITS.md).
- **Coarser units.** Merges still sit near word scale (mean length about 1.4 codes). An earlier ladder on a 100k-word slice missed its coarseness targets; those runs stay in the repo.
- **Scale and baselines.** WikiText-2 and a small model are the current proof. SentencePiece / unigram with a matched vocabulary, and anything web-scale, are not in this tree yet.

## Reproduce the working-scale run

Needs a GPU when `PRCF_REQUIRE_GPU=1` is set. Large model weights are not in git; this command rebuilds them.

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.working_lm.run
PYTHONPATH=src:. python -u -m experiments.working_lm.run --rescore-locality
```

[`experiments/working_lm/results.md`](experiments/working_lm/results.md)

### Earlier experiments

| Stage | Command |
|-------|---------|
| CodeBPE + smaller LM | `PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir6_bucket_codebpe.run_bakeoff` |
| Bucket+slot codec | `PYTHONPATH=src:. python -u -m experiments.dir5_bucket_slot.run_bench` |
| OOV hybrid decode | `PYTHONPATH=src:. python -u -m experiments.dir4_oov_decode.run_m1` |
| Scale ladder (gates missed) | `PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir3_scale_ladder.run_ladder --budget 30` |

Phases A–C (fingerprints, reconstruction, grouping) run on CPU. See each folder’s `results.md` under [`experiments/`](experiments/).
