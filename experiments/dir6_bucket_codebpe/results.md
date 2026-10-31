# Dir6 BucketCodeBPE + LM — **PASS**

- Device: `cuda`
- Stopped: `completed`
- Pass via: `lm_and_units`
- Elapsed: 405.8s

## Unit gates

- chrF: 1.0000 ok=True
- OOV exact: 1.0000 (n=6082) ok=True
- symbols/char: codebpe 0.2856 | raw bucket+slot 0.3343 | text BPE 0.3131 ok=True
- merges: 6070 (freq≥2: 6070, freq=1: 0) ok=True
- mean B unit len: 1.4786
- merge hist: {'freq_1': 0, 'freq_2_4': 5087, 'freq_5_9': 670, 'freq_10_plus': 313}

## LM bake-off

- Model: 3L/160d, 2000 steps
- BPC ratio (code/textBPE): 0.7632 (gate ≤ 1.05)
- LM wall: 24.5s

| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |
|---|---|---|---|---|---|
| BucketCodeBPE | 0.7474 | 96.85 | 0.4657 | 12387 | 2,920,320 / 928,160 |
| text BPE | 0.9792 | 466.88 | 0.3483 | 2000 | 1,258,400 / 928,160 |
