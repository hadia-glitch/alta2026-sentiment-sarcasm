# Sentiment & Sarcasm Classification Across English Varieties

Multi-task classification of sentiment and sarcasm in Australian and British English text, built for the ALTA 2026 Shared Task. The task pairs two signals that are notoriously entangled — sarcasm frequently inverts surface sentiment — and scores systems on their *weakest* language variety per task, rather than their average, which makes naive fine-tuning a poor fit and cross-group robustness a first-class design constraint rather than an afterthought.

## Results

Best submitted result (5-fold ensemble of `roberta-base` and a social-media-pretrained RoBERTa variant), verified on the official scorer:

| | sent-AU | sent-UK | sarc-AU | sarc-UK | **Final score** |
|---|---|---|---|---|---|
| Organizers' baseline (DistilBERT, single-task) | 0.9106 | 0.9532 | 0.7135 | 0.6374 | 0.7740 |
| **This project** | **0.9161** | **0.9377** | **0.7768** | **0.6493** | **0.7827** |

```
F1 score for sentiment (en-AU): 0.9161
F1 score for sentiment (en-UK): 0.9377
F1 score for sarcasm (en-AU):   0.7768
F1 score for sarcasm (en-UK):   0.6493
Final score:                    0.7827
```

The score is defined as `(min(sent-AU, sent-UK) + min(sarc-AU, sarc-UK)) / 2` — each task is graded on its weaker variety, so a model that does well on one dialect and poorly on the other is penalized as if it had failed both. Sentiment is close to solved by most systems on the public leaderboard; nearly the entire spread between entries is concentrated in sarcasm, where the positive-class rate ranges from ~30% (en-AU) down to ~8% (en-UK) — a genuinely imbalanced, low-resource sub-problem hiding inside what looks like a routine classification task.

A from-scratch classical baseline (TF-IDF + logistic regression, no pretrained language model) is included for comparison and reaches 0.7523 — useful as a fast, fully interpretable floor, and as a sanity check that the transformer pipeline's gains are real rather than an artifact of the evaluation harness.

## Data

The task uses the en-AU and en-UK subsets of [BESSTIE](https://github.com/dipankarsrirag/alta2026-sharedtask-baseline) (Google Places reviews and Reddit posts/comments), each row labelled with `source`, `variety`, `text`, `sentiment`, and `sarcasm`. Training data is 5,531 rows; the public development set is 757 rows.

Raw data files are included here for reproducibility. If the shared task's terms restrict redistribution, remove `data/train.csv` and `data/valid.csv` before publishing and pull them from the official source instead.

## Approach

**Baseline and feature analysis.** Before reaching for a pretrained model, a TF-IDF (word + character n-gram) logistic regression baseline established what a purely lexical model could and couldn't do. A handful of hand-engineered features — punctuation intensity, ALL-CAPS ratio, letter elongation, a lexicon-polarity mismatch signal — were tested head-to-head against the n-gram-only version and kept only where they measurably improved the weaker dialect's score, rather than assumed to help.

**Addressing class imbalance.** The sarcasm minority class (as low as ~8% positive in one dialect) was addressed on two fronts: a lightweight synthetic oversampling pass (targeted word deletion/swap) as a fast baseline technique, and an LLM-driven paraphrase augmentation pass targeting the scarcest slice specifically — prompting a language model to rewrite existing sarcastic examples in a different register while explicitly preserving the sarcasm, expanding the minority class with genuine lexical diversity rather than near-duplicates. Both were validated empirically before being folded into the training set, not assumed to help by default.

**Multi-task transformer fine-tuning.** A shared transformer encoder with two classification heads (sentiment, sarcasm) was fine-tuned across several pretrained backbones — `roberta-base`, `google/electra-base-discriminator`, a social-media-pretrained RoBERTa variant, and `microsoft/deberta-v3` — trained with mixed-precision (model weights kept in FP32, autocast handling the lower-precision forward pass) and gradient accumulation to fit within a single consumer GPU's memory budget. `deberta-v3` in particular required batch-level NaN/Inf detection, since its disentangled-attention mechanism is documented to occasionally spike to unstable gradients mid-fine-tune independent of the input data.

**Validation methodology.** Model selection and decision thresholds were tuned using 5-fold stratified cross-validation, with out-of-fold predictions as the only signal used for tuning — the labelled development set was reserved purely for reporting, never for any decision that could leak into the reported score. Where the training pool included augmented (paraphrased or synthetically perturbed) examples, fold splits were computed on the original data only, and each augmented example was constrained to the training side of whichever fold its source example belonged to — preventing a held-out fold from being evaluated against a near-duplicate of its own training data, which would otherwise inflate validation scores without any corresponding gain in generalization.

**Per-dialect calibration.** Because the scoring function grades en-AU and en-UK separately rather than pooling them, a single global decision threshold is provably suboptimal whenever the two varieties have different base rates (which they do, substantially, for sarcasm). Thresholds were instead calibrated independently per dialect against out-of-fold predictions, optimizing directly for the metric actually being scored rather than for accuracy or a pooled F1.

**Ensembling.** Final predictions blend probabilities across independently fine-tuned encoders. Averaging architecturally distinct models (different pretraining objectives and, in one case, a different pretraining domain) consistently outperformed any single model, including ones with a higher standalone validation score.

**Error analysis.** Manual inspection of misclassified examples on the harder, lower-resource dialect surfaced two consistent, distinct failure modes rather than random noise: sincere-but-intense language (strong opinions, profanity used earnestly) being mistaken for sarcasm, and genuinely deadpan sarcasm with no lexical or punctuation markers going undetected entirely. The two failure modes call for different fixes — the first is addressable through further threshold/calibration work, the second requires the kind of contextual reasoning a lexical model structurally cannot provide, which is what motivates the transformer approach in the first place.

**Reproducibility under constrained compute.** All transformer training checkpoints at both the fold level and the model level to persistent storage, so an interrupted training run resumes from the last completed fold rather than restarting from scratch — a meaningful concern when training runs happen on free-tier, time-limited GPU access.

## Repository structure

```
.
├── data/                 # train.csv, valid.csv, evaluate.py, sample_answer_template.csv
├── notebook/
│   └── sentiment_sarcasm_pipeline_refined.ipynb
├── submissions/          # dev_classical_baseline.zip, dev_multi_encoder_ensemble.zip
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_sarcasm_pipeline.ipynb
```

The classical baseline runs on CPU with no external dependencies beyond the packages above. The transformer pipeline requires a CUDA GPU and Hugging Face Hub access; it was developed and validated on a free-tier Colab GPU. The LLM-based augmentation step calls a hosted inference API and requires an API key set as an environment variable, everything else runs offline.

## Limitations & future directions

- Sarcasm on the lower-resource dialect remains the binding constraint on overall score.
- The current architecture shares a single encoder across both tasks with no auxiliary signal beyond the two primary loss terms. An earlier iteration explored a lightweight auxiliary head trained against a cheap heuristic proxy for sentiment/sarcasm inconsistency; it was dropped for this submission to keep memory usage within a single consumer GPU's budget, and is a candidate for revisiting given more compute.
- The per-dialect calibration currently tunes a decision threshold post hoc from out-of-fold probabilities. A more principled probability calibration step (e.g. temperature scaling per dialect) would likely improve the reliability of the predicted probabilities themselves, independent of the thresholding decision built on top of them.
- Cross-dialect generalization here concerns two varieties of the same language; extending the same calibration methodology to genuinely cross-lingual settings would require handling representation differences that don't arise between en-AU and en-UK.

## Acknowledgments

Task, data, and baseline courtesy of the ALTA 2026 Shared Task organizers and the [BESSTIE corpus / baseline repository](https://github.com/dipankarsrirag/alta2026-sharedtask-baseline).


