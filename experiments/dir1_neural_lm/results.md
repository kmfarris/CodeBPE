# Direction 1 — Neural LM bake-off — **PASS**

- Device: `cpu`
- Shared: 2L/128d, 800 steps, batch 16
- BPC ratio (span/BPE): 0.1876 (gate ≤ 1.05)

| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |
|---|---|---|---|---|---|
| span-PRCF | 0.5658 | 2801.61 | 0.0494 | 2051 | 667,520 / 396,800 |
| BPE | 3.0154 | 180.28 | 0.4024 | 500 | 468,992 / 396,800 |

Gate: span BPC ≤ BPE BPC × 1.05 on held-out paragraphs (equal architecture body; embedding tables scale with vocab).
