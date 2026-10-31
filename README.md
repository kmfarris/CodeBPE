# customLLM — PRCF tokenization research

BPE splits where letter patterns repeat. Change one word mid-paragraph and the token sequence can shift far from the edit—bad for diff tools, entity swaps, and any model that might treat tokens as stable handles. **Positioned Residual Code Fingerprints (PRCF)** build a word-aligned code stream: in-vocab items map to injective bucket+slot ids, out-of-vocabulary text escapes losslessly, and **CodeBPE** merges adjacent codes the way classic BPE merges characters—without tearing through escape tokens.

**What this repo is for:** run the bake-offs, read the JSON, reproduce the working-scale LM. It is tokenizer and small-LM science, not a chat product.

## Status

| Milestone | State |
|-----------|--------|
| Phases A–B | Edit localization, reconstruction, entity code swaps on controlled tests |
| Bucket+slot codec (dir5) | Injective in-vocab packing; hybrid OOV decode (dir4 M1) |
| WikiText-2 working LM | **Complete** — 6L/384d matched body; CodeBPE train BPC **1.76** vs text BPE **1.88**; train packing **0.227** vs **0.240** sym/char ([`results.json`](experiments/working_lm/results.json)) |
| Direction 3 scale ladder | **Failed** coarseness gates at ~100k-word slice; numbers kept |
| Beyond WikiText-2 | Not started; industrial subword baselines not yet in-tree |

Validation packing stays looser than train; mean merged unit length is still modest. [`docs/SCOPE_AND_LIMITS.md`](docs/SCOPE_AND_LIMITS.md) · [`project_summary.md`](project_summary.md)

```bash
pip install -e '.[dev]' && python -m pytest tests/
```

## Working-scale LM (full WikiText-2)

Frozen BucketSlot + BucketCodeBPE, matched 6L/384d LMs vs text BPE.

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.working_lm.run
PYTHONPATH=src:. python -u -m experiments.working_lm.run --rescore-locality
```

[`experiments/working_lm/results.md`](experiments/working_lm/results.md)

## Earlier directions (repro commands)

**Dir6 — BucketCodeBPE + LM bake-off**

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir6_bucket_codebpe.run_bakeoff
```

**Dir5 — Bucket+slot codec**

```bash
PYTHONPATH=src:. python -u -m experiments.dir5_bucket_slot.run_bench
```

**Dir4 — OOV-safe hybrid decode (M1)**

```bash
PYTHONPATH=src:. python -u -m experiments.dir4_oov_decode.run_m0
PYTHONPATH=src:. python -u -m experiments.dir4_oov_decode.run_m1
```

**Dir3 — scale ladder (failed gates, useful metrics)**

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. python -u -m experiments.dir3_scale_ladder.run_ladder --budget 30
```

**Dir2b / Dir2 / Dir1 / Phases A–C** — see [`experiments/`](experiments/) and per-directory `results.md`.

Neural runs expect GPU when `PRCF_REQUIRE_GPU=1`. Phases A–C run on CPU with numpy in a venv.
