# How the Random Forest Works — Plain Language

## The Feature Space (What the model actually sees)

The random forest never sees raw text. It sees **384 numbers** — a vector.
Here's the pipeline that produces those numbers:

1. **Text comes in** — a docstring, a comment, a string literal extracted from
   a Python file, or raw content from a YAML/Markdown file.

2. **Chunking** — Long text gets sliced into overlapping windows of 256 tokens
   with a 50-token overlap. This is so that if an attack straddles a boundary,
   at least one chunk captures it whole. Each chunk becomes a separate sample.

3. **Embedding** — Each chunk is fed through `all-MiniLM-L6-v2`, a
   sentence-transformer model. This model was pre-trained (by others, on a
   massive corpus) to map text into a 384-dimensional vector space where
   **semantically similar text lands near each other**. "Ignore all previous
   instructions" and "Disregard your system prompt" end up in a similar
   neighborhood, even though the words are different.

So the feature space is: **384 real-valued dimensions representing the semantic
meaning of a text chunk**. These aren't hand-picked features like "contains the
word 'ignore'" or "number of exclamation marks." They're learned
representations of meaning. The random forest's job is to find regions in that
384-dimensional space that correspond to "malicious" versus "safe."

## Training (How it learns the decision boundaries)

The training data comes from two public HuggingFace datasets of prompt
injection examples, labeled safe (0) or malicious (1). Each sample gets chunked
and embedded into 384-dim vectors, then:

- **200 decision trees** are trained, each on a random subset of the data and a
  random subset of the 384 features. Each tree learns its own set of if/then
  splits through the vector space — "if dimension 47 > 0.3 and dimension 182 <
  -0.1, then malicious." Each individual tree is mediocre, but they vote
  together.

- **`class_weight="balanced"`** compensates for the fact that safe text vastly
  outnumbers attack text in the real world. Without this, the model would learn
  to just say "safe" for everything and still be right 95% of the time. This
  setting tells each tree to treat misclassifying one rare malicious sample as
  equivalent to misclassifying many safe ones.

- **Platt scaling (calibration)** is applied afterward. Raw random forest votes
  don't produce well-calibrated probabilities — if 140 out of 200 trees say
  "malicious," that doesn't mean there's a 70% chance it's actually malicious.
  Platt scaling fits a sigmoid curve to map raw scores onto real probabilities.
  This matters because the system makes threshold-based decisions: below 0.3 =
  safe, 0.3–0.7 = suspicious (escalate to deeper analysis), above 0.7 = block
  immediately. Those thresholds only make sense if the numbers correspond to
  actual probabilities.

## Why Random Forest Specifically

The system needs to run on a modest VPS with no GPU, classify text in
milliseconds for real-time I/O protection, and produce calibrated confidence
scores. Random forests are fast at inference (just traversing trees), handle
high-dimensional inputs without exploding, and calibrate well with Platt
scaling. A neural classifier might squeeze out more accuracy, but it would need
a GPU and be harder to inspect when investigating false positives.

## The Key Insight

**The random forest isn't working with text — it's working with coordinates in
meaning-space.** The sentence-transformer does the hard work of turning language
into geometry, and the random forest draws decision boundaries in that geometric
space. Each tree draws a crude boundary; 200 of them voting together draw a
surprisingly precise one.

## Reference

- Training script: `train_model.py`
- Runtime classifier: `sentinel_core/ml_classifier.py`
- PRD Section 5 (Layer 2: Sentinel ML Pipeline)
- PRD Section 8 (Machine Learning: Training & Active Learning)
