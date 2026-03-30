# Deep Past Initiative — Akkadian Machine Translation

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview)
[![Kaggle Model](https://img.shields.io/badge/Kaggle-AkkaByte--GapFixed-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/models/sudharsananh/akkabyte-gapfixed/PyTorch/default/1)
[![Train Notebook](https://img.shields.io/badge/Notebook-Training_v9-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/sudharsananh/fork-of-vesudeep/edit/run/302359743)
[![Inference Notebook](https://img.shields.io/badge/Notebook-Inference_v9-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ragunathravi/fork-of-big-gap-fix-inference/edit/run/302454888)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](LICENSE.txt)

---

A complete end-to-end pipeline for translating ancient Old Assyrian (Akkadian) cuneiform tablets from the Kultepe/Kanesh archive into English. The system spans automated publication mining using a 120-billion-parameter language model, lexicon-augmented data engineering, ByT5 fine-tuning with model weight averaging, and Minimum Bayes Risk (MBR) decoding at inference time.

---

## Architecture

![System Architecture](img/architecture.svg)

The pipeline operates across five stages: raw scholarly data ingestion, GPU-accelerated publication mining, multi-source data engineering with gloss augmentation, ByT5 sequence-to-sequence fine-tuning with model soup, and MBR-guided inference with structured postprocessing.

---

## Problem Statement

The Deep Past Initiative challenge tasks participants with automatically translating Akkadian cuneiform transliterations from Old Assyrian merchant tablets (circa 1950–1750 BCE) into English. These tablets, excavated from the ancient trading colony of Kanesh in modern-day Turkey, record commercial contracts, caravan logistics, family correspondence, and debt agreements.

The core difficulty is threefold. First, the source language is morphologically complex and highly abbreviated in scholarly transliteration. Second, training data is scarce: only approximately 1,500 full document pairs exist in the competition dataset. Third, damaged or illegible tablet sections appear as structured gap markers that must be preserved exactly rather than translated.

---

## Solution Overview

### Stage 1 — Data Sources

The pipeline draws from six distinct data sources, all combined and deduplicated before training:

**Scholarly Publications.** A corpus of PDFs from major Old Assyrian publication series including AKT, BIN, CCT, ICK, EL, TC, TPAK, KTS, KTH, VS, and several others published by cuneiform scholars. These contain professional English, French, German, and Turkish translations of tablets not present in the competition training set.

**Competition CSVs.** The official `train.csv` (document-level pairs), `test.csv` (inference targets), and `Sentences_Oare_FirstWord_LinNum.csv` (sentence-span alignments within documents).

**OA Lexicon eBL.** A structured lexicon mapping Akkadian surface forms and normalized forms to their lexeme identifiers across 54,511 entries. Used to build gloss annotations during data engineering.

**eBL Dictionary.** Maps lexemes to English gloss strings via the `definition` column. Combined with the OA Lexicon, this produces a full chain: Akkadian word form to English meaning.

**Published Texts.** Additional tablet texts from the broader eBL corpus providing transliteration material.

**Extracted AKT Pairs.** A curated set of approximately 211 translation pairs extracted from AKT publication PDFs using the publication mining pipeline described in Stage 2.

---

### Stage 2 — Publication Mining (`code/publications_datamining.py`)

Because the competition training set contains only 1,545 document pairs, a secondary extraction pipeline was built to mine professional translations directly from scanned scholarly PDFs.

**GPU Smart Matching.** Stage 1 of the extraction (upstream of this file) uses GPU-batched OR-combined regular expressions to identify which pages of which PDFs contain references to specific tablet IDs. It accounts for Roman-to-Arabic numeral variants (BIN 4 = BIN IV), punctuation variants between series, volume, and text number, and sublabel suffixes (A, B, obv., rev.).

A literal sanity check filters false positives: a page flagged by a GPU batch may have matched a different ID in the same batch. After this filter, pages are capped at five per tablet ID, preferring longer pages by character count.

**120B LLM Extraction.** Each qualifying page is sent to a 120-billion-parameter model (GPT-OSS 120B) running under vLLM on an H100 GPU. The model receives a detailed system prompt encoding domain knowledge: publication series abbreviations, how translations appear in interlinear scholarly layout, the distinction between valid prose translations and bibliographic citations or raw Akkadian transliterations, and language handling rules for EN / FR / DE / TR / AK output.

The model is instructed to reason step-by-step through six explicit steps before emitting a single JSON object as its final output. This chain-of-thought approach reduces hallucination on edge cases such as interlinear text mixing Akkadian and English lines on alternating rows.

The inference runs at temperature 0.0 for deterministic output. Context is set to 16,384 tokens with FP8 KV cache (`fp8_e4m3`) to maximize batch concurrency on the H100. A tiered fallback system trims page text progressively from full to 24,000 characters to 12,000 characters if a batch triggers a context overflow.

**Post-processing Filters.** After extraction, results go through Akkadian contamination detection (more than four hyphenated syllabic tokens or Sumerian logogram patterns in the English output triggers rejection), OCR artifact correction (subscript digits, fused ligatures, broken hyphens, stray citation references), fuzzy deduplication at a 90 percent similarity threshold using SequenceMatcher, and a minimum word count filter of six words per translation.

Non-English translations (French, German, Turkish) have their `original_text_found` preserved and their `english_translation` produced by the same model in the same extraction call, with explicit translation instructions in the system prompt.

A checkpoint-and-resume mechanism saves parsed outputs every five batches so a long extraction run can restart without reprocessing completed rows.

---

### Stage 3 — Data Engineering (`code/train.py`)

All training data is constructed inline within the training script, removing the dependency on pre-built intermediate CSVs.

**Lexicon Construction.** The OA Lexicon and eBL Dictionary are loaded and merged in two steps. First, `OA_Lexicon_eBL.csv` maps surface forms and normalized forms to their lexeme identifiers, applying `ḫ/Ḫ → h/H` substitution and subscript digit normalization. Second, `eBL_Dictionary.csv` maps lexemes to English gloss strings via the `definition` column (a critical fix over prior versions that searched for a non-existent `english` column). The combined lexicon contains 54,511 entries. A secondary accent-normalized lookup is generated by NFD-decomposing keys and stripping combining marks, providing a fallback for accented Akkadian forms.

**Data Source 1: Document Pairs.** Each row in `train.csv` yields one base pair and, if the gloss injection threshold is met, one augmented pair. Preprocessing strips determinative parentheses, unifies gap markers, removes brackets while preserving their content, and applies the character normalization maps. Validity filtering rejects pairs where the target has fewer than three words or the source-to-target length ratio exceeds four-to-one for longer texts. This yields approximately 1,545 original pairs and a similar count of glossed variants, controlled by a minimum match ratio of 0.15 (15 percent of unique source words must hit the lexicon before glosses are injected).

**Data Source 2: Sentence Pairs.** The sentence CSV aligns first-word indices to tablet-level transliterations, allowing sentence spans to be extracted from the full document. Each span and its corresponding translation forms a pair. The minimum match ratio for gloss injection is 0.08, accounting for the shorter average token count in sentence-level pairs. This yields approximately 820 pairs.

**Data Source 3: Extracted AKT Pairs.** The 211 pairs from Stage 2 are loaded, filtered against a blacklist of known bad tablet IDs, stripped of commentary patterns (texts beginning with phrases like "Note concerning" or "Account of"), deduplicated by target text, and preprocessed identically to the document pairs. The minimum gloss injection ratio is 0.03, reflecting that proper names and Sumerograms dominate extracted tablet identifiers. Approximately 175 glossed variants are generated from this source.

**Data Source 4: Proper Noun Pairs.** A small synthetic set of up to 200 pairs is generated from the lexicon's personal name (PN) and geographical name (GN) entries using fixed Akkadian address formula templates (greeting clauses, witness lines, father-son constructions). This teaches the model canonical transliteration patterns for common names. The set is intentionally capped: earlier experiments with 1,500 proper noun pairs (39 percent of total training data) caused the leaderboard score to regress from 36 to 32, as synthetic templates overwhelmed real training signal.

**Final Assembly.** All four sources are concatenated. Exact (source, target) duplicates are removed. A 10 percent validation split is drawn from sentence pairs only (to avoid leaking document-level translations into both splits), with a floor of 50 pairs. Training pairs are shuffled with a fixed seed. The final training set contains approximately 6,600 pairs versus 4,937 in prior versions.

---

### Stage 4 — Model Training (`code/train.py`)

**Base Model.** Training begins from `byt5-akkadian-optimized-34x`, a ByT5 checkpoint that has already been adapted to Akkadian transliteration. ByT5 operates on raw UTF-8 byte sequences rather than subword tokens, which is critical for a low-resource language with heavy use of diacritics (š, ṭ, ṣ, ḥ, ā, ī, ū) and specialized romanization conventions. There is no out-of-vocabulary problem and no need for vocabulary extension.

**Training Configuration.** The model is trained for up to 20 epochs with a batch size of 24, gradient accumulation of 2 (effective batch size 48), a learning rate of 1e-5, cosine decay with 5 percent warmup, Adafactor optimizer, and BF16 mixed precision on the H100. Maximum sequence length is 512 tokens. Early stopping monitors the combined metric (geometric mean of BLEU and chrF++) with a patience of 5 evaluation steps. Evaluation and checkpointing occur every 100 steps.

**Evaluation Metrics.** The combined metric is defined as the geometric mean of corpus BLEU and corpus chrF with word order 2. This penalizes models that sacrifice fluency for adequacy or vice versa. Checkpoint selection is driven entirely by this combined metric on the sentence-level validation split.

**Model Soup.** After fine-tuning, the best checkpoint is linearly interpolated with the public baseline checkpoint at the tensor level: 70 percent fine-tuned weights and 30 percent public baseline weights. This weight-averaging technique (model soup) consistently improves held-out performance by reducing overfitting to the training distribution. A safety routine re-ties `embed_tokens` to `shared.weight` for any checkpoint where they were not saved as physical clones, preventing silent embedding corruption from old checkpoints.

The tokenizer is always loaded from the public baseline rather than the fine-tuned checkpoint, because some fine-tuned tokenizer configs have `extra_special_tokens` serialized as a list rather than a dict, which breaks newer versions of the Transformers library.

---

### Stage 5 — Inference (`code/inference.py`)

**Model Loading.** The inference script attempts models in priority order: soup checkpoint, then fine-tuned final checkpoint, then public baseline as a fallback. Each candidate is validated by generating a translation for a fixed Akkadian test phrase and checking that the output is coherent printable English (more than 3 characters, at least 80 percent printable ASCII-range characters).

**Preprocessing and Gloss Injection.** Each test transliteration goes through the same preprocessing as training: determinative normalization, gap unification across more than 20 orthographic variants (angle brackets, ellipses, bracketed x sequences, explicit break annotations), bracket stripping with content preservation, and character normalization. Glosses are then injected from the lexicon using the same injection logic as training, with the same minimum coverage threshold.

**MBR Decoding.** Each input is decoded using a pool of candidates generated from two inference passes. The first pass runs beam search with 12 beams (10 for short inputs under 60 tokens), returning all beam hypotheses. The second pass runs ancestral sampling at temperature 0.7, top-p 0.92, returning 3 sequences. The candidate pool is capped at 32 after deduplication.

Minimum Bayes Risk selection scores each candidate by computing its average chrF++ (word-order-2) similarity against all other candidates in the pool. The candidate maximizing this expected utility is selected as the final translation. This approach consistently outperforms simple beam search argmax by preferring consensus outputs over individually high-scoring but idiosyncratic generations.

**Postprocessing.** The selected translation goes through a structured postprocessing pipeline. Gap tokens are protected via a sentinel byte sequence before any character stripping occurs, then restored, preventing gap markers from being corrupted by the forbidden-character translation step. Soft grammatical markers (feminine, plural, uncertain annotations) are removed. Curly quotation marks are converted to straight quotes rather than deleted. The string `'s` is rewritten to `s` (possessive apostrophes). Month names are converted from Roman to Arabic numerals. Repetitions are collapsed at both the word level and the phrase level (up to 6-gram repeated spans). Fractions such as 0.3333 or 1.3333 are left untouched as floats, matching competition label conventions confirmed by the host.

BF16 autocasting is applied during inference when the GPU supports it, approximately halving memory usage and improving throughput with no measurable quality loss.

---

## Repository Structure

```
.
├── code/
│   ├── inference.py                   Inference pipeline: MBR decoding, postprocessing, submission
│   ├── train.py                       Data engineering + ByT5 fine-tuning + model soup
│   └── publications_datamining.py     120B LLM extraction from scholarly PDFs
├── img/
│   └── architecture.svg              System architecture diagram
├── output/
│   └── submission.csv                Sample submission output
├── README.md
└── LICENSE.txt                        GNU General Public License v3
```

---

## Training Data Statistics

| Source | Type | Pairs (Base) | Pairs (Glossed) | Gloss Threshold |
|---|---|---|---|---|
| train.csv | Document | ~1,545 | ~1,545 | 0.15 |
| Sentences CSV | Sentence | ~820 | ~810 | 0.08 |
| AKT Publications | Extracted | ~211 | ~175 | 0.03 |
| OA Lexicon PN/GN | Proper Noun | ~200 | 0 | N/A |
| **Total** | | **~2,776** | **~2,530** | |
| **Combined** | | **~6,600 after dedup** | | |

---

## Hyperparameter Reference

| Parameter | Value | Notes |
|---|---|---|
| Base model | byt5-akkadian-optimized-34x | ByT5 byte-level encoder-decoder |
| Epochs | 20 | Early stopping patience 5 |
| Batch size | 24 | Per device, gradient accum 2 |
| Learning rate | 1e-5 | Cosine decay, 5% warmup |
| Optimizer | Adafactor | No momentum parameters |
| Precision | BF16 | H100 native |
| Max sequence length | 512 | Tokens (bytes for ByT5) |
| Soup ratio | 70/30 | Fine-tuned / public baseline |
| MBR beam count | 12 | 10 for inputs under 60 tokens |
| MBR sample count | 3 | Temperature 0.7, top-p 0.92 |
| MBR pool cap | 32 | After deduplication |
| Length penalty | 1.6 | Beam search |
| Repetition penalty | 1.2 | Beam search and sampling |

---

## Key Implementation Notes

**The lexicon column name fix.** Prior training versions searched for an `english` or `English` column in `eBL_Dictionary.csv`, which does not exist. The actual column is named `definition`. This caused the lexicon to fall back to bare lexeme strings rather than English glosses in all runs before v9, silently producing zero effective glossed training pairs despite the 54,511-entry lexicon appearing to load successfully.

**No fraction conversion.** Early versions of the postprocessing pipeline converted fractions like 1/3 to decimals. Competition host confirmation established that test labels contain floats such as 0.3333 and 1.3333 directly, so all fraction conversion logic was removed.

**Parentheses preservation.** The character stripping step explicitly excludes parentheses because they appear in test labels as part of legitimate translation content (e.g., speculative readings, uncertain readings). Removing them caused systematic mismatches against reference translations.

**Proper noun cap.** Experiments with 1,500 proper noun synthetic pairs caused a leaderboard regression from 36 to 32 because the synthetic templates constituted approximately 39 percent of total training data, overwhelming the signal from real translation pairs. The cap was reduced to 200 pairs.

**Tokenizer source.** The tokenizer is always loaded from the public baseline checkpoint regardless of which model weights are used. Fine-tuned checkpoint tokenizer configs serialized `extra_special_tokens` as a Python list rather than a dict in some versions, causing crashes in HuggingFace Transformers >= 4.40.

---

## Requirements

**Hardware.** A single NVIDIA H100 80GB GPU is recommended for both the publication mining stage (120B model inference) and training (ByT5 fine-tuning). Training can run on a single A100 40GB with reduced batch size. Inference requires at minimum a GPU with 24GB VRAM.

**Software Dependencies.**

```
torch>=2.0
transformers>=4.40
datasets
sacrebleu
vllm
pandas
numpy
scikit-learn
tqdm
langdetect
```

**Data.** Competition data must be accessible at the paths defined in the `COMP_DIR` constant in `train.py`. The publication PDFs and GPU match outputs from Stage 1 (not included in this repository) are required for the mining pipeline.

---

## Running the Pipeline

**Data engineering and training:**

```bash
python code/train.py
```

The script reads competition CSVs, builds the lexicon, constructs all training sources, trains the model, and writes the soup checkpoint to `akkadian-byt5-model-v9/soup/`.

**Publication mining (optional, improves extracted pair count):**

```bash
python code/publications_datamining.py
```

Requires `gpu_matches_smart.csv` from an upstream GPU matching run and the 120B model weights accessible at `MODEL_PATH`.

**Inference and submission generation:**

```bash
python code/inference.py
```

Reads `test.csv`, runs MBR decoding, applies postprocessing, and writes `submission.csv`.

---

## Results

| Metric | Value |
|---|---|
| Evaluation metric | chrF++ (word order 2) |
| Avg words per translation | 25.4 |
| Training pairs (v9) | ~6,600 |
| Training pairs (v8, baseline) | 4,937 |
| Empty translations filled | Fallback to "The tablet is too damaged to translate." |

---

## References

The following scholarly works and resources underpin the data and lexical resources used in this pipeline:

Veenhof, K.R. and Eidem, J. (2008). *Mesopotamia: The Old Assyrian Period.* Fribourg/Gottingen.

Michel, C. (2001). *Correspondance des marchands de Kanish.* Paris: Le Cerf.

eBL (electronic Babylonian Library) project. Freie Universitat Berlin. https://www.ebl.lmu.de/

Larsen, M.T. (1976). *The Old Assyrian City-State and its Colonies.* Copenhagen.

Cecen, S. and Donbaz, V. (1995). *AKT I: Anadolu Kaynaklarinda Ticaret.* Ankara.

Hecker, K. et al. (1998). *Kappadokische Keilschrifttafeln aus den Sammlungen der Karlsuniversitat Prag.* Praha.

---

## License

This project is released under the GNU General Public License v3.0. See [LICENSE.txt](LICENSE.txt) for full terms.
