# Path 2 + scaled bake-off — **PASS**

- Units: CodeBPE (merges=600, vocab=4912, mean unit len=1.13 codes)
- Round-trip chrF train/test: 0.9346/0.9380
- Model: 3L/160d, 1000 steps (scaled vs dir1 2L/128d/800)
- BPC ratio (code/textBPE): 0.5164 (gate ≤ 1.05 and chrF≥0.80)

| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |
|---|---|---|---|---|---|
| CodeBPE | 1.6588 | 12107.94 | 0.1223 | 4912 | 1,724,320 / 928,160 |
| text BPE | 3.2125 | 253.18 | 0.4024 | 500 | 1,018,400 / 928,160 |
