# Working-scale BucketCodeBPE LM — **PASS**

- Device: `cuda`
- Data train/valid words: 2005825 / 209190
- Elapsed: 19559.3s

## Packing

- CodeBPE train/valid sym/char: 0.2269 / 0.2682
- text BPE train/valid sym/char: 0.2404 / 0.2422

## Invertibility smoke: chrF=1.0000 oov_exact=1.0000

## LM bake-off

- Model: 6L/384d, 20000 steps, batch=8
- BPC ratio: 0.9380 (gate ≤ 1.05) ok=True
- LM wall: 1164.1s

| substrate | BPC ↓ | PPL ↓ | valid sym/char | vocab | params (all / non-embed) |
|---|---|---|---|---|---|
| BucketCodeBPE | 1.7635 | 14.58 | 0.2682 | 59117 | 33,446,784 / 10,647,552 |
| text BPE | 1.8800 | 17.41 | 0.2422 | 10000 | 14,585,856 / 10,647,552 |

## Locality

- ±2 acc: 1.0000 (n=300, gate ≥ 0.8) ok=True
- exact: 1.0000
- index space: split_codec (gold aligned to codec whitespace tokens)

Samples: `samples.txt`
