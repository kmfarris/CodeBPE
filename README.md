# PRCF — codes as an alternative to normal tokens

## Working-scale LM (full WikiText-2) — PASS

Full WikiText-2 (~2.0M / 209k words), frozen BucketSlot + BucketCodeBPE, matched 6L/384d LMs vs text BPE.

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.working_lm.run
# locality only (after codecs + results exist):
PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.working_lm.run --rescore-locality
```

See [`experiments/working_lm/results.md`](experiments/working_lm/results.md).

## Dir6 — BucketCodeBPE + LM bake-off — PASS

Merges adjacent bucket+slot ids only (never through ESC). Train packing beats text BPE; matched 3L/160d LM beats text BPE on BPC.

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.dir6_bucket_codebpe.run_bakeoff
```

See [`experiments/dir6_bucket_codebpe/results.md`](experiments/dir6_bucket_codebpe/results.md).

## Dir5 — Bucket+slot codec — PASS

In-vocab words get injective `(RQ_bucket, slot)` packed as one id (no river/rider smash). OOV/punct still ESC+chars.

```bash
PYTHONPATH=src:. .venv/bin/python -u -m experiments.dir5_bucket_slot.run_bench
```

See [`experiments/dir5_bucket_slot/results.md`](experiments/dir5_bucket_slot/results.md).

## Dir4 — OOV-safe hybrid decode — M1 PASS

Legacy PRCF nearest-word decode fails held-out chrF (~0.65). HybridCodec uses RQ for canonical in-vocab words and ESC+chars for OOV/collisions.

```bash
PYTHONPATH=src:. .venv/bin/python -u -m experiments.dir4_oov_decode.run_m0
PYTHONPATH=src:. .venv/bin/python -u -m experiments.dir4_oov_decode.run_m1
```

See [`experiments/dir4_oov_decode/results_m1.md`](experiments/dir4_oov_decode/results_m1.md).

## Dir3 — Scale ladder (WikiText-2) — budget 30 FAIL (useful)

~100k-word train slice, frequency-stable CodeBPE (`allow_freq1=False`), matched tiny LM vs text BPE.

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.dir3_scale_ladder.run_ladder --budget 30
```

See [`experiments/dir3_scale_ladder/results.md`](experiments/dir3_scale_ladder/results.md).

## Dir2b — Coarse CodeBPE + GPU bake-off — PASS (toy; freq=1 caveat)

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.dir2b_coarse_units.run_bakeoff
```

## Direction 2 — mild CodeBPE baseline — PASS

```bash
PRCF_REQUIRE_GPU=1 PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.dir2_code_bpe.run_bakeoff
```

## Direction 1 — Neural LM bake-off — PASS

```bash
PYTHONPATH=src:. /home/kelly/miniconda3/bin/python -u -m experiments.dir1_neural_lm.run_bakeoff
```

## Phases A–C (foundation)

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.phase_a.run_bench
PYTHONPATH=src:. .venv/bin/python -m experiments.phase_b.run_bench
PYTHONPATH=src:. .venv/bin/python -m experiments.phase_c.run_bench
```

Neural runs need conda Python + GPU outside sandbox (`PRCF_REQUIRE_GPU=1`). `.venv` has numpy for A–C.

See [`project_summary.md`](project_summary.md) for findings and [`docs/SCOPE_AND_LIMITS.md`](docs/SCOPE_AND_LIMITS.md) for metric scope. Tests: `pip install -e '.[dev]' && python -m pytest tests/`.
