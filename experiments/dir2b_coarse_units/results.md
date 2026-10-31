# Dir2b coarse CodeBPE — **PASS**

- Device: `cuda`
- Stopped: `completed`
- Pass via: `lm_and_units`

## Unit gates

| check | measured | gate | ok |
|---|---|---|---|
| mean unit len | 2.0000 | ≥ 2.0 | True |
| symbols/word | 0.4934 | ≤ 0.65 | True |
| chrF test | 0.9380 | ≥ 0.9 | True |

- Merges: 6445 (freq≥2: 600, freq=1: 5845)
- Vocab: 3658, max unit len: 17

## LM bake-off

- Model: 3L/160d, 1000 steps
- BPC ratio (code/textBPE): 0.3593 (gate ≤ 1.05)

| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |
|---|---|---|---|---|---|
| CodeBPE | 1.1542 | 101926.90 | 0.0743 | 3658 | 1,523,680 / 928,160 |
| text BPE | 3.2125 | 253.18 | 0.4024 | 500 | 1,018,400 / 928,160 |
