# Project summary: an alternative to standard tokenization

**Project:** CodeBPE / PRCF (Positioned Residual Code Fingerprints)  
**Question:** If we rethink tokenization — what we cut text into before a language model sees it — what do we gain?

The purpose is to measure that substrate: structured codes instead of usual BPE/WordPiece pieces, and a small language model trained on those codes.

---

## What we set out to learn

Ordinary tokenizers split text by spelling patterns. We tested a different pipeline:

1. Can a representation of text keep edits **local** (change one word → only nearby pieces change)?
2. Can those pieces be **swapped** (e.g. replace a name’s codes) and still decode back to readable text?
3. Can **learned, coarser units** (short phrases of codes) work as the alphabet a small language model predicts — compared with standard subword tokens under matched training?

---

## Findings (short)

| Stage | Question | Result |
|---|---|---|
| Phase A | Can a positioned fingerprint find which word changed? | **Yes** — exact recovery on synthetic one-word edits. Collapsed hashes (SimHash) cannot. |
| Phase B | Can learned per-word codes reconstruct, localize edits, and support entity swaps? | **Yes** — high reconstruction quality; edit localization and name/city code transplants worked on our tests. |
| Phase C / Dir1 | Do grouped codes help as an LM alphabet vs BPE? | **Yes on bits-per-character** in small bake-offs — but early “glue every 3 words” grouping was arbitrary. |
| Dir2 | Learn merge rules over word-codes (CodeBPE)? | **Yes on BPC**, but units stayed almost word-sized (mean length ~1.13). |
| Dir2b | Force merges until units average ~2 codes? | **Yes on toy data** — mean length 2.0, but most merges were freq=1 (not frequency-stable). |
| Dir3 (30 min) | Same idea on ~100k WikiText-2 words, **freq≥2 only**? | **FAIL gates, useful numbers** — packing win (0.11 vs 0.31 sym/char), mean unit len only **1.48**, held-out chrF **0.64** (nearest-word OOV), BPC ratio **1.05**. |
| Dir4 M0/M1 | Fix decode with hybrid RQ + ESC/chars? | **M1 PASS** — chrF/OOV exact **1.00**; packing **worse** than BPE (0.44 vs 0.31) — collision losers spelled out. |
| Dir5 | Bucket+slot (injective in-vocab codes)? | **PASS** — collision rate **0**, chrF/OOV exact **1.00**, packing **0.33** vs BPE 0.31; bucket-path **80%/69%**. |
| Dir6 | CodeBPE on bucket ids + LM vs text BPE? | **PASS** — 6070 freq≥2 merges, train sym/char **0.286** (≤ raw 0.334 and ≤ BPE 0.313), chrF/OOV exact **1.00**, BPC **0.747** vs text BPE **0.979** (ratio **0.76**). |
| Working LM | Full WikiText-2 + 6L/384d equal-body bake-off? | **PASS** — invertibility **1.00**, train packing **0.227** vs BPE **0.240**, BPC **1.76** vs **1.88** (ratio **0.94**), locality ±2 **1.00** (gold in codec/`split()` index space). |

**Bottom line so far:** On full WikiText-2, injective bucket+slot codes plus BB-only CodeBPE keep **invertibility and edit locality**, pack denser than text BPE on train, and win **BPC** under a matched 6L/384d LM. Web-scale remains open.

---

## Possible benefits of rethinking tokenization (plain language)

If the cut of the text changes, the benefits we care about here are about **how the sequence is packaged**, not whether the model “knows more”:

- **Locality:** A one-word change shows up in a small region of the code sequence. Useful for diffs, caches, and “what changed?” tooling.
- **Editability:** Swap codes for a name or short phrase and decode — patch the representation without retokenizing the whole string from scratch.
- **Coarser units:** Fewer sequence steps for the same paragraph (more content per piece). That can mean shorter contexts for the same text, or denser packing of content into a fixed window.
- **A different tradeoff than spelling-BPE:** BPE optimizes for frequent letter patterns. Code merges can optimize for frequent *adjacent codes* (closer to repeated phrases), which is a different design axis for a tokenizer.

Those are the hypothesized payoffs of an alternative tokenizer. Whether they win at large scale is still open.

---

## Limitations and open questions

1. **Valid packing still looser than train.** Working LM: CodeBPE valid sym/char **0.268** vs train **0.227** (and vs BPE valid **0.242**) — more ESC/OOV on held-out.
2. **Not web-scale** — full WikiText-2 + 6L/384d is a working substrate proof, not a production LM.
3. **Mean merge coarseness still mild** — working LM mean B-unit len **~1.40** after 20k freq≥2 merges (target 1.5).
4. **Toy Dir2b coarseness was partly artificial** (freq=1 merges); later runs use freq≥2 only.
5. **Locality scoring must use codec index space** — bare `tokenize()` gold vs `split()` streams falsely failed (0.39); remapped gold → **1.00**.
6. **First Dir6 unit attempt failed** (OOV E-atoms → UNK); fixed by lossless reconstruct + ESC expansion in LM encode — not a silent pass.

**Next:** SentencePiece or unigram with a matched vocabulary on the same split; then larger data and stronger coarseness if those baselines still favor PRCF.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Token** | A chunk of text in the sequence a model reads/writes. |
| **Tokenization** | How raw text is cut into (and rebuilt from) those chunks. |
| **BPE / WordPiece** | Common spelling-based ways to build a token vocabulary from data. |
| **Tokenizer** | The encode/decode system: text ↔ token IDs. |
| **Language model (LM)** | Predicts the next item in a sequence (here: used to compare alphabets). |
| **Fingerprint** | Here: an ordered sequence of codes for a paragraph (not a password-style hash). |
| **Cryptographic hash (e.g. SHA)** | Any tiny input change randomizes the digest; cannot localize which word changed. |
| **Positioned fingerprint** | Aligned to places in the text so edits stay local. |
| **SimHash / MinHash** | Similarity digests; nearby texts match, but one digest usually won’t name the changed word. |
| **Shingle / n-gram** | Overlapping short phrases; one word change only hits nearby shingles. |
| **Embedding** | A list of numbers standing for a piece of text. |
| **Vector quantization (VQ)** | Map a number-list to the nearest codebook entry → a discrete **code**. |
| **Residual VQ (RQ)** | Stacked codebooks: coarse code, then codes for the leftover error. |
| **PRCF** | Positioned residual code fingerprints — word-aligned RQ codes plus tooling. |
| **Code / code tuple** | Discrete ID(s) for a word or merged phrase after quantization. |
| **Decode / reconstruction** | Codes → text. |
| **chrF** | How closely reconstructed text matches the original (1.0 = perfect). |
| **CodeBPE** | BPE-style merges over *adjacent codes* (not letters) to form longer units. |
| **Mean unit length** | Average word-codes inside one LM piece. ~1 ≈ word-sized; ~2 ≈ short phrases. |
| **Symbols per word** | LM pieces per word of text (lower = coarser tokenization). |
| **Bits per character (BPC)** | Surprise per character under the LM; lower is better. Useful when comparing different token alphabets. |
| **Perplexity (PPL)** | Surprise per symbol *of that alphabet*; not comparable across different tokenizers. |
| **Bake-off** | Same LM body/budget, different tokenizers, compare metrics. |
| **Transformer** | Standard neural LM architecture used in the bake-offs. |

---

## Where the code lives

| Path | Role |
|---|---|
| [`src/prcf/`](src/prcf/) | Fingerprints, RQ, BucketSlotCodec, BucketCodeBPE, tiny LM |
| [`experiments/phase_a/`](experiments/phase_a/) … [`phase_c/`](experiments/phase_c/) | Classical → learned → span foundations |
| [`experiments/dir1_neural_lm/`](experiments/dir1_neural_lm/) | First neural codes vs BPE |
| [`experiments/dir2_code_bpe/`](experiments/dir2_code_bpe/) | Mild CodeBPE baseline |
| [`experiments/dir2b_coarse_units/`](experiments/dir2b_coarse_units/) | Coarse units + GPU bake-off (toy PASS; freq=1 caveat) |
| [`experiments/dir3_scale_ladder/`](experiments/dir3_scale_ladder/) | WikiText-2 scale ladder (**budget 30 FAIL**) |
| [`experiments/dir4_oov_decode/`](experiments/dir4_oov_decode/) | Hybrid decode (**M1 PASS**) |
| [`experiments/dir5_bucket_slot/`](experiments/dir5_bucket_slot/) | Bucket+slot (**PASS**) |
| [`experiments/dir6_bucket_codebpe/`](experiments/dir6_bucket_codebpe/) | BucketCodeBPE + LM (**PASS**) |
| [`README.md`](README.md) | How to re-run experiments |

---

## One-sentence takeaway

**Bucket+slot + BB-only CodeBPE** is an invertible tokenizer alternative that, at small WikiText scale, **packs denser than text BPE on train and wins BPC** under a matched tiny LM.
