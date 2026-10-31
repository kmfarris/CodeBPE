# Dir3 scale ladder — **FAIL**

- Budget: `30`
- Device: `cuda`
- Stopped: `in_progress`
- Data: train_words=100013 valid_words=20067

## Stage 15 — tokenizer (freq-stable CodeBPE)

- Result: **FAIL** (`chrf_fail`)
- chrF valid: 0.6411 (gate ≥ 0.85)
- symbols/char: CodeBPE 0.1102 vs text BPE 0.3131
- mean unit len: 1.4757
- merges: 6655 (freq≥2: 6655, freq=1: 0)
- merge hist: {'freq_1': 0, 'freq_2_4': 5578, 'freq_5_9': 744, 'freq_10_plus': 333}
- walls: {'data': 4.849747707834467, 'prcf': 4.172166822012514, 'code_bpe': 49.160654358100146, 'stage15_total': 200.9568588770926}
