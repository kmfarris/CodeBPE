# Evaluation scope and limitations

Every number in this repository sits inside a specific experimental contract. This page states that contract: what the artifacts establish, what they do not, and how to read the metrics without mixing configurations.

## What the record supports

**Reconstructibility and locality.** On WikiText-2-scale bucket+slot codes, single-word edits remain localized in code space, and decode returns readable text. The working-scale run documents invertibility and ±2 locality in codec index space (`experiments/working_lm/results.json`).

**Matched language models.** Under a fixed body (six layers, 384-dimensional hidden states), a CodeBPE alphabet over bucket ids achieves lower train bits-per-character than text BPE on the same corpus split. That is a controlled comparison between alphabets—not a claim about production-scale perplexity.

**Direction 3.** The scale ladder failed its predefined coarseness gates. Those runs remain in the tree because negative results constrain what the merge policy can promise at intermediate scale.

## What lies outside current evidence

Web-scale and multilingual tokenization. Inference economics against industrial subword toolchains at billion-token scale. Phrase-level compression: mean merged unit length stays near word scale (~1.4 codes after frequency-filtered merges).

## Reading the metrics

**Train and validation packing move together but not identically.** CodeBPE validation symbols-per-character exceed train when held-out text triggers escape and out-of-vocabulary paths. Report both columns from `experiments/working_lm/results.json`.

**Vocabulary scale differs by direction.** Early experiments used smaller vocabs; `working_lm` uses the full WikiText-2 word inventory. Do not merge tables without noting the configuration.

**Bucket-path rates are not a single headline ratio.** Direction 5 reports bucket-path coverage alongside BPE packing on comparable settings. Injective in-vocab coding improves reconstructibility; aggregate packing gains depend on corpus and merge budget.

**Locality labels live in codec space.** Gold edit regions are defined on `split()` streams. Scoring against raw `tokenize()` output produced false negatives in early runs; the remapped protocol is documented in `project_summary.md`.

**Checkpoints stay out of git.** Large `.pt` weights are gitignored. Regenerate with `python -m experiments.working_lm.run` (see root `README.md`).

## Baselines a formal write-up would still add

SentencePiece or unigram tokenization with matched vocabulary on the same WikiText-2 split. A single script emitting validation packing beside train metrics for every direction cited in prose.

## Artifacts

Gate definitions, commands, and numeric outputs: `experiments/*/results.json` and companion `results.md` files.
