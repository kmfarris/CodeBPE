# Phase C results — **PASS**

- Pass via: `modeling_match`
- Train/test paragraphs: 955/168
- Design: LM atoms = hashed groups of token-RQ codes; decode uses token RQ

## Modeling (equal-compute trigram LM)

| substrate | BPC ↓ | symbols/char ↓ | vocab |
|---|---|---|---|
| span-grouped token RQ (size=3) | 0.7049 | 0.0494 | 4099 |
| BPE | 2.6889 | 0.4024 | 500 |

- BPC ratio (span/BPE): 0.2622 (gate ≤ 1.05, requires chrF)
- Compression ratio (span/BPE symbols): 0.1228 (gate ≤ 0.55)

## Quality / editability

| metric | value | gate |
|---|---|---|
| reconstruct chrF | 0.9380 | ≥ 0.8 |
| phrase localize ±1 | 1.0000 | ≥ 0.7 |
| phrase swap acc | 0.8500 | (informational) |
