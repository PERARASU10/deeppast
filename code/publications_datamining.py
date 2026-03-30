# %% [code] {"execution":{"iopub.status.busy":"2026-02-19T08:25:17.772604Z","iopub.execute_input":"2026-02-19T08:25:17.773830Z","iopub.status.idle":"2026-02-19T08:28:51.534898Z","shell.execute_reply.started":"2026-02-19T08:25:17.773808Z","shell.execute_reply":"2026-02-19T08:28:51.534358Z"},"jupyter":{"outputs_hidden":false}}
!pip install -q openai-harmony vllm

# %% [code] {"execution":{"iopub.status.busy":"2026-02-19T08:28:51.535901Z","iopub.execute_input":"2026-02-19T08:28:51.536048Z","iopub.status.idle":"2026-02-19T08:28:59.740209Z","shell.execute_reply.started":"2026-02-19T08:28:51.536031Z","shell.execute_reply":"2026-02-19T08:28:59.739729Z"},"jupyter":{"outputs_hidden":false}}
!pip install -q langdetect

# %% [code] {"execution":{"iopub.status.busy":"2026-02-19T09:25:52.948547Z","iopub.execute_input":"2026-02-19T09:25:52.948766Z","iopub.status.idle":"2026-02-19T09:57:14.277229Z","shell.execute_reply.started":"2026-02-19T09:25:52.948751Z","shell.execute_reply":"2026-02-19T09:57:14.276758Z"},"jupyter":{"outputs_hidden":false}}
"""
=============================================================================
Stage 2: High-Quality Translation Extraction Pipeline — v2
120B Model | Akkadian → English | Deep Past Challenge
=============================================================================
Design principles:
  - Full page text always sent — no fragile context window clipping
  - Model thinks step-by-step BEFORE outputting JSON (chain-of-thought)
  - Pre-dedup: literal sanity check + cap 5 pages per oare_id
  - Post-dedup: fuzzy similarity dedup + Akkadian contamination filter
  - Handles:
      EN           → extract as-is
      FR/DE/TR/... → extract original + translate to English
      AK           → original_text_found set, english_translation = NONE
  - Output columns:
      oare_id, original_id, source_pdf,
      status,
      original_text_found,     <- exact text as found in publication
      original_language_code,  <- EN / FR / DE / TR / AK / OTHER
      english_translation,     <- English prose, or NONE if Akkadian
      confidence,              <- HIGH / MEDIUM / LOW
      is_truncated,            <- True if translation appears cut off
      reason                   <- only on INVALID rows
=============================================================================
"""

import pandas as pd
import re
import json
import os
from collections import Counter
from difflib import SequenceMatcher

from vllm import LLM, SamplingParams
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 0.  CONFIGURATION  —  edit these before running
# =============================================================================

MATCHES_CSV       = "/kaggle/input/datasets/sudharsananh/publicationsv2/gpu_matches_smart.csv"   # Stage 1 output
OUTPUT_CSV        = "stage2_extracted_v2.csv"
MODEL_PATH        = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"   # <-- UPDATE THIS

# Checkpoint settings — saves raw outputs periodically so you can resume
# if the kernel crashes or times out
CHECKPOINT_FILE   = "stage2_checkpoint.csv"      # raw parsed rows saved here during inference
CHECKPOINT_EVERY  = 5                             # save checkpoint every N batches
RESUME_FROM_CHECKPOINT = False                    # True = skip already-processed rows on restart

TEST_MODE         = False      # True = 500-sample test run; False = full run
TEST_SAMPLE_SIZE  = 500       # number of unique oare_ids to sample in test mode

BATCH_SIZE        = 16        # H100 80GB can handle larger batches
MAX_NEW_TOKENS    = 8192      # 8k output tokens — enough for full CoT + JSON across all tiers
TEMPERATURE       = 0.0       # greedy / deterministic — important for extraction

MAX_PAGES_PER_ID  = 5         # max pages per oare_id sent to the model
FUZZY_THRESHOLD   = 0.90      # deduplicate if >90% similar (raised from 85% — catches OCR variants of same text)
MIN_WORD_COUNT    = 6         # drop translations under 6 words — too short for MT training

# H100-specific GPU settings
TARGET_GPU_MEMORY_UTILIZATION = 0.95
KV_CACHE_DTYPE                = "fp8_e4m3"   # fp8 KV cache — H100 native, saves ~50% KV memory
# 16k context (not 32k) — critical for throughput on a single H100:
#   32k → only 3.9x concurrency (3.3 GiB KV cache, ~4 seqs in flight)
#   16k → ~7.8x concurrency (same KV cache, sequences are half the size)
# 95%+ of scholarly OCR pages fit within 16k tokens. Full text is still sent.
MAX_MODEL_LEN                 = 16384

# Fallback tiers — tried in order if a batch fails (context overflow / OOM)
# max_page_chars=None means send full page unchanged (fits within 16k)
# max_new_tokens kept at 8192 across all tiers — user-set for chain-of-thought quality
# Tiers only trim the INPUT page text to fit smaller context windows:
#   Tier 1: full page,    up to 16k total context
#   Tier 2: page ~24k chars (~8k token input)  + 8k output = ~16k total
#   Tier 3: page ~12k chars (~4k token input)  + 8k output = ~12k total
CONTEXT_TIERS = [
    {"max_page_chars": None,   "max_new_tokens": 8192},  # Tier 1: full page, 16k context
    {"max_page_chars": 24_000, "max_new_tokens": 8192},  # Tier 2: trimmed,    8k input
    {"max_page_chars": 12_000, "max_new_tokens": 8192},  # Tier 3: trimmed,    4k input
]


# =============================================================================
# 1.  SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a specialist assistant for the Deep Past Initiative, 
a scholarly project that mines academic publications to recover English translations 
of ancient Old Assyrian (Akkadian) cuneiform tablets from the Kültepe/Kanesh archive.

Your sole task: given a full page from an OCR-processed scholarly publication, 
find and extract the translation of a specific target tablet ID.

===========================================================================
DOMAIN KNOWLEDGE — apply this when analysing the page text
===========================================================================

1. PUBLICATION SERIES (tablets are cited using these abbreviations):
   BIN, CCT, ICK, EL, TC, TPAK, AKT, Kt, KTS, KTH, VS, OAA, POAT, CTMMA,
   CMK, BT, PLbg, KEL, AAA, TTC, ATHE, KKS, JTVI, Donbaz, Hecker, Matous,
   Dercksen, Veenhof, Ichisar, Larsen, Ulshöfer, Prag, Kayseri

2. ID VARIANT FORMS — the same tablet may be cited in many ways:
   "BIN 4 189" = "BIN IV 189" = "BIN, 4, no.189" = "BIN IV, 189" = "B.I.N.4.189"
   - Arabic ↔ Roman numerals for volume numbers: 4=IV, 6=VI, 1=I, 2=II etc.
   - Spacing, commas, periods between series / volume / text number vary freely
   - A sublabel may appear: "BIN 4 189 A", "ICK 2 95 A/B", "Kt 91/k 123"
   - OCR corruptions are common: "l" for "1", "0" for "O", "fi" fused, broken hyphens

3. HOW TRANSLATIONS APPEAR in scholarly publications:
   - They follow the Akkadian transliteration on the same page or nearby
   - They may be: an indented block, inline after the ID, a block quote
   - Interlinear format: Akkadian line, then English line, alternating
   - They are introduced by phrases like: "Translation:", "Traduction:", 
     "Übersetzung:", "Çeviri:", "'...'", "«...»", "(transl.)", 
     "The text reads:", "We may translate:", "We read:", "(trans.)"
   - Footnotes and commentaries CITE the ID without giving a translation —
     these are NOT valid translations

4. PUBLICATION LANGUAGES: translations appear in English (en), French (fr),
   German (de), or Turkish (tr). All are valuable — extract + translate them.

5. WHAT COUNTS AS A VALID TRANSLATION:
   VALID:
     ✓ Prose with real subject + verb (even if fragmentary or partial)
     ✓ "... [broken] ... he shall pay 2 minas of silver within one month"
     ✓ Only one side of the tablet (obverse or reverse) — still valid
     ✓ Business content: debts, caravans, loans, family letters, contracts
   NOT VALID:
     ✗ A list of personal names with no sentence
     ✗ A pure commodity list with only numbers (e.g., "3 talents tin, 2 donkeys")
     ✗ A bibliography or footnote that only cites the tablet without translating
     ✗ The Akkadian transliteration itself (hyphenated syllables: "a-na be-li-ni")
     ✗ Scholarly commentary discussing the text linguistically

===========================================================================
CRITICAL LANGUAGE RULES — follow these exactly
===========================================================================

ENGLISH (original_language_code = "EN"):
  - original_text_found = the exact text from the publication
  - english_translation = same as original_text_found

FRENCH / GERMAN / TURKISH / OTHER modern language (original_language_code = "FR"/"DE"/"TR"/"OTHER"):
  - original_text_found = the exact text as it appears in the publication
  - english_translation = your accurate English translation of that text
  - Do NOT skip non-English translations — they contain unique, valuable data
  - Translate faithfully and completely; do not summarise or paraphrase

AKKADIAN (original_language_code = "AK"):
  - This happens when only the Akkadian transliteration is present, with no
    modern-language translation visible anywhere on the page
  - original_text_found = the Akkadian syllabic text as found
  - english_translation = NONE   ← NEVER attempt to translate Akkadian;
    that is exactly what this competition is asking models to learn to do
  - status = INVALID with reason "Only Akkadian transliteration present, no translation"

===========================================================================
OCR ARTEFACT HANDLING
===========================================================================
The page was produced by OCR on a scanned PDF. You will encounter:
  - Random mid-sentence line breaks (join them naturally)
  - Garbled diacritics: ā ī ū š ṣ ṭ ḥ (Akkadian characters corrupted)
  - Merged words: "ofsilver" → "of silver", "heshall" → "he shall"
  - Stray page numbers, headers, footers embedded mid-text (skip them)
  - Lightly fix obvious OCR joins/splits in your output but do NOT rewrite
    or paraphrase the translation — preserve the scholarly wording

===========================================================================
OUTPUT FORMAT — strictly required
===========================================================================
First, reason through ALL steps below explicitly.
Then, as the VERY LAST THING in your response, output exactly one JSON object.

JSON schema:
{
  "status": "VALID" or "INVALID",
  "original_text_found": "<the exact text from the publication in its original language>",
  "original_language_code": "EN" | "FR" | "DE" | "TR" | "AK" | "OTHER",
  "english_translation": "<clean English prose>" or "NONE",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "is_truncated": true or false,
  "reason": "<brief reason — REQUIRED only if INVALID, else null>"
}

JSON rules:
  - All strings must be properly JSON-escaped (escape internal quotes with \\")
  - english_translation must be the string "NONE" when original_language_code = "AK"
  - is_truncated = true if the translation appears cut off mid-sentence
  - confidence levels:
      HIGH   → ID found clearly, translation is unambiguous complete prose
      MEDIUM → ID found but translation is partial, ambiguous, or needs repair
      LOW    → ID located via variant/OCR guess only; translation uncertain
  - If status = VALID, reason should be null
"""


# =============================================================================
# 2.  USER PROMPT TEMPLATE
# =============================================================================

def build_user_prompt(target_id: str, raw_text: str) -> str:
    return f"""## TARGET TABLET ID
`{target_id}`

Variant forms to also look for (same tablet, different citation styles):
  - Arabic ↔ Roman volume numbers: e.g. "BIN 4" also appears as "BIN IV"
  - Punctuation variants: spaces, commas, periods between series/volume/number
  - Sublabels: "{target_id} A", "{target_id} B", "{target_id} obv."
  - OCR corruptions: "l" vs "1", "0" vs "O", merged ligatures

---

## FULL PAGE TEXT FROM OCR'D PUBLICATION

{raw_text}

---

## REASONING STEPS — work through each one explicitly before writing your JSON

STEP 1 — FIND THE ID
  Scan the entire page text above.
  Does `{target_id}` (or any of its variants listed above) appear anywhere?
  Quote the exact sentence or line where you found it.
  If it is completely absent from the page text, immediately output INVALID 
  with reason "Target ID not found in text" and stop.

STEP 2 — CLASSIFY WHAT FOLLOWS THE ID
  Look at the text immediately surrounding where you found the ID.
  Classify it as ONE of:
    (a) An English translation of the tablet → proceed to Step 3
    (b) A French / German / Turkish / other modern-language translation → Step 3
    (c) The Akkadian transliteration (hyphenated syllables) with a modern-language
        translation also visible on the page → extract the translation in Step 3
    (d) Only Akkadian with NO modern-language translation visible → INVALID (AK)
    (e) A bibliography entry or footnote citation only (no translation) → INVALID

STEP 3 — EXTRACT THE TRANSLATION
  Copy the translation text EXACTLY as it appears in the publication.
  - If it is interlinear (Akkadian line / English line alternating), collect 
    ONLY the modern-language translation lines and join them into one block.
  - If it continues across a line break, join the lines naturally.
  - Include the full text — do not truncate it.
  - If the page cuts off mid-sentence, include what is there and set 
    is_truncated = true.
  - Do NOT include the Akkadian lines, line numbers, footnote markers, 
    or page headers in original_text_found.

STEP 4 — DETECT LANGUAGE
  What language is the extracted text in?
  - If English: english_translation = same as original_text_found
  - If French/German/Turkish/Other: translate it accurately to English.
    Preserve the meaning, names, and technical terms; do not summarise.
  - If Akkadian: english_translation = NONE

STEP 5 — QUALITY CHECK
  Does this pass the VALID test? (real prose with subject + verb)
  Is it accidentally contaminated with Akkadian syllables?
  Is it truncated? What confidence level is appropriate?

STEP 6 — OUTPUT JSON
  Now write the final JSON object as the very last thing in your response.
  No text after the closing brace.
"""


# =============================================================================
# 3.  PRE-PROCESSING — clean gpu_matches_smart.csv before inference
# =============================================================================

def roman_arabic_variants(term: str) -> list:
    """Generate common Roman↔Arabic numeral swaps for a term."""
    variants = [term]
    pairs = [('1','I'),('2','II'),('3','III'),('4','IV'),('5','V'),
             ('6','VI'),('7','VII'),('8','VIII'),('9','IX'),('10','X')]
    for arab, roman in pairs:
        v1 = re.sub(rf'\b{arab}\b', roman, term)
        v2 = re.sub(rf'\b{roman}\b', arab, term)
        if v1 != term: variants.append(v1)
        if v2 != term: variants.append(v2)
    return variants


def literal_match(row) -> bool:
    """Return True if matched_term (or a variant) appears in raw_text_fragment."""
    term = str(row.get('matched_term', '')).strip()
    text = str(row.get('raw_text_fragment', '')).lower()
    for variant in roman_arabic_variants(term):
        if variant.lower() in text:
            return True
    return False


def preprocess_matches(path: str, test_mode: bool, sample_size: int) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("PRE-PROCESSING: gpu_matches_smart.csv")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"Loaded raw rows:       {len(df):,}")

    # -- Literal sanity check --------------------------------------------------
    # GPU batch used OR-combined regex: a page may have been flagged by a
    # different ID in the batch, not the actual target. Filter these out.
    before = len(df)
    df['_literal_ok'] = df.apply(literal_match, axis=1)
    df = df[df['_literal_ok']].drop(columns=['_literal_ok'])
    print(f"After literal check:   {len(df):,}  (removed {before - len(df):,} false positives)")

    # -- Dedup same oare_id + same source_pdf ----------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=['oare_id', 'source_pdf'], keep='first')
    print(f"After (id+pdf) dedup:  {len(df):,}  (removed {before - len(df):,})")

    # -- Cap MAX_PAGES_PER_ID per oare_id, prefer longer pages ----------------
    df['_tlen'] = df['raw_text_fragment'].fillna('').str.len()
    df = df.sort_values(['oare_id', '_tlen'], ascending=[True, False])
    df = df.groupby('oare_id').head(MAX_PAGES_PER_ID).reset_index(drop=True)
    df = df.drop(columns=['_tlen'])
    print(f"After {MAX_PAGES_PER_ID} pages/id cap:   {len(df):,}  ({df['oare_id'].nunique():,} unique oare_ids)")

    # -- Test mode sampling ----------------------------------------------------
    if test_mode:
        # Sample diverse oare_ids evenly
        unique_ids = df['oare_id'].unique()
        # Sample min(sample_size, available unique ids)
        n = min(sample_size, len(unique_ids))
        import numpy as np
        np.random.seed(42)
        chosen_ids = np.random.choice(unique_ids, size=n, replace=False)
        df = df[df['oare_id'].isin(chosen_ids)].reset_index(drop=True)
        print(f"\n[TEST MODE] Sampled {len(df):,} rows across {df['oare_id'].nunique()} oare_ids")

    return df


# =============================================================================
# 4.  JSON PARSING — robust, handles model thinking before JSON
# =============================================================================

REQUIRED_FIELDS = [
    'status', 'original_text_found', 'original_language_code',
    'english_translation', 'confidence', 'is_truncated'
]

def parse_model_output(raw_output: str) -> dict:
    """
    Find and parse the last JSON object in the model's response.
    The model is instructed to think first, output JSON last.
    """
    # Strategy 1: find all {...} blocks, try from last one backward
    # Use a greedy approach that handles nested braces
    brace_starts = [m.start() for m in re.finditer(r'\{', raw_output)]
    for start in reversed(brace_starts):
        # Find matching closing brace
        depth = 0
        for i, ch in enumerate(raw_output[start:]):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = raw_output[start : start + i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if all(f in parsed for f in REQUIRED_FIELDS):
                            return parsed
                    except json.JSONDecodeError:
                        # Try mild fixes: remove literal newlines inside strings
                        try:
                            fixed = re.sub(r'(?<=[^\\])\n', ' ', candidate)
                            parsed = json.loads(fixed)
                            if all(f in parsed for f in REQUIRED_FIELDS):
                                return parsed
                        except:
                            pass
                    break

    # Strategy 2: field-by-field regex extraction as last resort
    result = {
        'status': 'PARSE_ERROR',
        'original_text_found': None,
        'original_language_code': None,
        'english_translation': None,
        'confidence': 'LOW',
        'is_truncated': False,
        'reason': f'JSON parse failed. Raw output snippet: {raw_output[-300:]}'
    }
    for field in ['status', 'original_language_code', 'confidence']:
        m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', raw_output)
        if m:
            result[field] = m.group(1)
    return result


# =============================================================================
# 5.  POST-PROCESSING
# =============================================================================

AK_SYLLABLE_RE  = re.compile(r'\b[a-záéíúšṣṭḥ]{1,5}-[a-záéíúšṣṭḥ]{1,5}\b')
SUMERIAN_LOG_RE = re.compile(r'\b[A-ZÁÉÍÚŠṢṬḤ]{2,}\.[A-ZÁÉÍÚŠṢṬḤ]{2,}\b')

def is_akkadian_contaminated(text: str) -> bool:
    """Detect if english_translation accidentally contains Akkadian."""
    if not text or text == 'NONE':
        return False
    ak_hits = AK_SYLLABLE_RE.findall(text.lower())
    if len(ak_hits) > 4:  # occasional loanwords OK; many = contamination
        return True
    if SUMERIAN_LOG_RE.search(text):
        return True
    return False


def light_ocr_clean(text: str) -> str:
    """
    OCR cleanup for english_translation:
      - joins hyphen line-breaks
      - strips footnote number refs like "6) stands at..."
      - strips inline scholar citations like "OIP 27 55:10, var. from..."
      - strips subscript OCR artifacts like lu2, ki3
      - normalises whitespace
    """
    if not text or text == 'NONE':
        return text

    # Join hyphenated line-breaks (word-hy-\nphen → wordhy phen → wordhyphen)
    text = re.sub(r'-\n\s*', '', text)

    # Strip inline scholar citation references (e.g. "BIN 6 124:10; abi ina q...")
    # Pattern: CAPS series + numbers + colon + number = citation leak
    text = re.sub(r'\b[A-Z]{2,}[\s\d]+:\d+[^.]*', '', text)

    # Strip footnote number markers mid-sentence: "6) stands at" → "stands at"
    # Only when digit+paren appear after whitespace (not at start of sentence)
    text = re.sub(r'(?<=\s)\d+\)\s*', '', text)
    # Also strip at very start of text
    text = re.sub(r'^\d+\)\s*', '', text)

    # Strip OCR subscript artifacts: lu2, ki3, id2 etc (lowercase + single digit)
    text = re.sub(r'\b([a-záéíúšṣṭḥ]{2,})\d\b', r'\1', text)

    # Normalise whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', ' ', text)
    text = re.sub(r'\n', ' ', text)

    # Drop stray line-number markers at very start (e.g., "1' 2' ")
    text = re.sub(r"^[\d'\" ]+", '', text)

    return text.strip()


def fuzzy_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:600], b[:600]).ratio()


CONF_RANK = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

def postprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("POST-PROCESSING")
    print(f"{'='*60}")
    print(f"Total rows in:  {len(df):,}")

    valid   = df[df['status'] == 'VALID'].copy()
    invalid = df[df['status'] != 'VALID'].copy()
    print(f"VALID:   {len(valid):,}")
    print(f"INVALID/ERROR: {len(invalid):,}")

    # FIX 1 — Remove Akkadian-contaminated english_translation ------------------
    before = len(valid)
    valid = valid[~valid['english_translation'].apply(
        lambda x: is_akkadian_contaminated(str(x))
    )]
    print(f"After Akkadian contamination filter: {len(valid):,}  (removed {before - len(valid):,})")

    # FIX 2 — OCR cleanup (footnotes, citations, subscripts, whitespace) --------
    valid['english_translation'] = valid['english_translation'].apply(
        lambda x: light_ocr_clean(str(x)) if x and x != 'NONE' else x
    )
    valid['original_text_found'] = valid['original_text_found'].apply(
        lambda x: light_ocr_clean(str(x)) if x else x
    )

    # FIX 3 — Drop translations too short to be useful for MT training ----------
    valid['_wcount'] = valid['english_translation'].apply(
        lambda x: len(str(x).split()) if x and x != 'NONE' else 0
    )
    before = len(valid)
    valid = valid[valid['_wcount'] >= MIN_WORD_COUNT]
    print(f"After min-word-count filter (>={MIN_WORD_COUNT}): {len(valid):,}  (removed {before - len(valid):,})")

    # FIX 4 — Fuzzy dedup with compound key: oare_id + original_id -------------
    # Groups by BOTH to handle cases where one oare_id has multiple original_ids
    # (aliased tablets). Within each compound group, dedup by text similarity.
    # Then across compound groups for the same oare_id, keep best by word count.
    valid['_crank'] = valid['confidence'].map(CONF_RANK).fillna(0)
    valid['_tlen']  = valid['_wcount']   # word count is better than char count
    valid = valid.sort_values(['oare_id', 'original_id', '_crank', '_tlen'],
                              ascending=[True, True, False, False])

    # Step 4a: dedup within each (oare_id, original_id) compound group ----------
    kept = []
    for (oare_id, orig_id), group in valid.groupby(['oare_id', 'original_id']):
        accepted = []
        for _, row in group.iterrows():
            trans  = str(row.get('english_translation', ''))
            is_dup = any(
                fuzzy_sim(trans, str(a.get('english_translation', ''))) > FUZZY_THRESHOLD
                for a in accepted
            )
            if not is_dup:
                accepted.append(row.to_dict())
        kept.extend(accepted)

    compound_deduped = pd.DataFrame(kept)
    print(f"After fuzzy dedup ({FUZZY_THRESHOLD:.0%}) within (oare_id+original_id): "
          f"{len(compound_deduped):,}")

    # Step 4b: for same oare_id with multiple original_ids, keep best per oare_id
    # "Best" = highest word count (most complete translation), tie-break by confidence
    if len(compound_deduped) > 0:
        compound_deduped = compound_deduped.sort_values(
            ['oare_id', '_crank', '_tlen'], ascending=[True, False, False]
        )
        # Keep ALL rows — multiple distinct original_ids for same oare_id can
        # represent genuinely different translations from different publications.
        # We only collapse if they are the SAME original_id (done in step 4a).
        # Flag oare_ids that have multiple original_ids for review.
        id_counts = compound_deduped.groupby('oare_id')['original_id'].nunique()
        multi_id  = id_counts[id_counts > 1].index
        compound_deduped['oare_id_aliased'] = compound_deduped['oare_id'].isin(multi_id)
        aliased_count = compound_deduped['oare_id_aliased'].sum()
        if aliased_count > 0:
            print(f"  WARNING: {len(multi_id)} oare_ids have multiple original_ids "
                  f"(aliased tablets) — flagged in 'oare_id_aliased' column")

    deduped = compound_deduped.copy() if len(compound_deduped) > 0 else pd.DataFrame()
    if len(deduped) > 0:
        deduped = deduped.drop(columns=['_crank', '_tlen', '_wcount'], errors='ignore')
    print(f"Final valid rows: {len(deduped):,}  "
          f"({deduped['oare_id'].nunique() if len(deduped) > 0 else 0} unique oare_ids)")

    final = pd.concat([deduped, invalid], ignore_index=True)
    return final


# =============================================================================
# 6.  MAIN PIPELINE
# =============================================================================

def main():

    # ---- Pre-process --------------------------------------------------------
    df = preprocess_matches(MATCHES_CSV, TEST_MODE, TEST_SAMPLE_SIZE)

    # ---- Resume: detect already-processed rows from checkpoint ---------------
    already_done_indices = set()
    checkpoint_rows      = []

    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            ckpt = pd.read_csv(CHECKPOINT_FILE)
            checkpoint_rows = ckpt.to_dict('records')
            # The checkpoint stores df row indices so we know exactly which to skip
            if '_df_index' in ckpt.columns:
                already_done_indices = set(ckpt['_df_index'].dropna().astype(int).tolist())
            print(f"\n[RESUME] Found checkpoint: {CHECKPOINT_FILE}")
            print(f"[RESUME] Already processed: {len(already_done_indices):,} rows — will skip these")
        except Exception as e:
            print(f"[RESUME] Could not load checkpoint ({e}) — starting fresh")
            checkpoint_rows      = []
            already_done_indices = set()
    else:
        print(f"\n[RESUME] No checkpoint found — starting fresh")

    # ---- Build chat prompts (only for unprocessed rows) ----------------------
    print(f"\nBuilding prompts for {len(df):,} rows...")
    chat_messages    = []   # prompts to run
    pending_indices  = []   # df row index for each prompt (for checkpoint tracking)

    for idx, row in df.iterrows():
        if idx in already_done_indices:
            continue        # skip — already in checkpoint
        target_id = str(row.get('matched_term', row.get('original_id', 'UNKNOWN'))).strip()
        raw_text  = str(row.get('raw_text_fragment', ''))
        chat_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(target_id, raw_text)}
        ])
        pending_indices.append(idx)

    print(f"Prompts to run:  {len(chat_messages):,}  "
          f"(skipped {len(already_done_indices):,} already-done rows)")

    # ---- Load model — auto GPU detection + context fallback -----------------
    import torch
    import time

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs detected — cannot run inference.")
    print(f"\nDetected {n_gpus} GPU(s). Loading model: {MODEL_PATH}")

    # Try context sizes from largest to smallest until one fits in VRAM.
    context_fallbacks = [MAX_MODEL_LEN, 8192, 4096]  # 16k → 8k → 4k
    llm = None
    loaded_ctx = None

    for ctx in context_fallbacks:
        try:
            print(f"  Trying max_model_len={ctx} ...")
            llm = LLM(
                model=MODEL_PATH,
                dtype="auto",
                kv_cache_dtype=KV_CACHE_DTYPE,
                gpu_memory_utilization=TARGET_GPU_MEMORY_UTILIZATION,
                max_model_len=ctx,
                tensor_parallel_size=n_gpus,
                trust_remote_code=True,
                # enforce_eager intentionally omitted — lets vLLM use CUDA graph capture
                # which gives 2-3x throughput vs eager mode (was the main speed bottleneck)
            )
            loaded_ctx = ctx
            print(f"  ✓ Loaded with context {ctx}")
            break
        except Exception as e:
            print(f"  ✗ Failed at context {ctx}: {str(e)[:120]}")
            llm = None
            time.sleep(2)

    if llm is None:
        raise RuntimeError("Model failed to load at all context sizes. Check VRAM / model path.")

    # Filter tiers: Tier 1 (full page / None) is only valid if loaded context is big enough.
    # With 16k loaded, Tier 1 sends full page which must fit in 16k — that's fine for most pages.
    # Any tier whose max_page_chars would require more than loaded_ctx is skipped.
    # Rough rule: 1 token ≈ 4 chars, so max safe chars for a context = loaded_ctx * 4 * 0.6 (60% for input)
    max_safe_chars = loaded_ctx * 4 * 0.6   # ~38k chars for 16k ctx

    active_tiers = []
    for t in CONTEXT_TIERS:
        mpc = t["max_page_chars"]
        if mpc is None or mpc <= max_safe_chars:
            active_tiers.append(t)
    if not active_tiers:
        active_tiers = [CONTEXT_TIERS[-1]]   # always have at least the smallest tier

    # ---- Helper: truncate page text to fit a context tier -------------------
    def truncate_page(raw_text: str, max_chars) -> str:
        """Trim page text to max_chars, cutting at nearest sentence boundary."""
        if max_chars is None or len(raw_text) <= max_chars:
            return raw_text
        cutpoint = raw_text.rfind('. ', 0, max_chars)
        if cutpoint == -1:
            cutpoint = max_chars
        return raw_text[:cutpoint + 1] + "\n[PAGE TRIMMED TO FIT CONTEXT WINDOW]"

    def apply_tier_to_messages(messages, max_page_chars):
        """Return a copy of the chat messages with page text trimmed if needed."""
        if max_page_chars is None:
            return messages
        result = []
        for turn in messages:
            if turn["role"] == "user":
                content = turn["content"]
                m = re.search(
                    r"(## FULL PAGE TEXT FROM OCR'D PUBLICATION\n+)(.*?)(\n+---)",
                    content, re.DOTALL
                )
                if m:
                    new_page = truncate_page(m.group(2), max_page_chars)
                    content  = content[:m.start(2)] + new_page + content[m.end(2):]
                result.append({"role": "user", "content": content})
            else:
                result.append(turn)
        return result

    # ---- Inference: Tier 1 → Tier 2 → Tier 3 with checkpoint saves ----------
    print(f"\nRunning inference — batches={BATCH_SIZE} | "
          f"max_new_tokens={MAX_NEW_TOKENS} | "
          f"loaded_ctx={loaded_ctx} | active tiers: {len(active_tiers)}")
    print(f"Checkpoint: saving every {CHECKPOINT_EVERY} batches → {CHECKPOINT_FILE}")

    raw_outputs      = []
    tier_usage       = {i+1: 0 for i in range(len(active_tiers))}
    batches_since_ckpt = 0

    BATCH_ERROR_TEMPLATE = (
        '{{"status":"BATCH_ERROR","original_text_found":null,'
        '"original_language_code":null,"english_translation":null,'
        '"confidence":"LOW","is_truncated":false,'
        '"reason":"all context tiers failed — {err}"}}'
    )

    def save_checkpoint(raw_outputs_so_far, pending_indices_so_far,
                        df, checkpoint_rows_from_prev):
        """
        Parse outputs collected so far, merge with df metadata,
        combine with any rows loaded from a previous checkpoint, and save.
        """
        if not raw_outputs_so_far:
            return
        # Parse what we have
        parsed_now = []
        for out in raw_outputs_so_far:
            parsed_now.append(parse_model_output(out))

        # Build meta for the pending rows processed so far
        indices_so_far = pending_indices_so_far[:len(parsed_now)]
        meta_rows = df.loc[indices_so_far, ['oare_id', 'matched_term', 'source_pdf']].copy()
        meta_rows = meta_rows.rename(columns={'matched_term': 'original_id'})
        meta_rows['_df_index'] = indices_so_far   # track row index for resume

        parsed_df = pd.DataFrame(parsed_now)
        new_ckpt_rows = pd.concat(
            [meta_rows.reset_index(drop=True), parsed_df.reset_index(drop=True)],
            axis=1
        )

        # Combine with any previously loaded checkpoint rows
        if checkpoint_rows_from_prev:
            prev_df   = pd.DataFrame(checkpoint_rows_from_prev)
            combined  = pd.concat([prev_df, new_ckpt_rows], ignore_index=True)
        else:
            combined  = new_ckpt_rows

        combined.to_csv(CHECKPOINT_FILE, index=False)
        return len(combined)

    for batch_num, i in enumerate(
        tqdm(range(0, len(chat_messages), BATCH_SIZE), desc="Extracting")
    ):
        batch_orig  = chat_messages[i : i + BATCH_SIZE]
        batch_out   = None
        last_err    = "unknown"

        for tier_num, tier in enumerate(active_tiers, start=1):
            sp = SamplingParams(
                temperature=TEMPERATURE,
                max_tokens=tier["max_new_tokens"],
            )
            try:
                batch_tiered = [
                    apply_tier_to_messages(msgs, tier["max_page_chars"])
                    for msgs in batch_orig
                ]
                results   = llm.chat(batch_tiered, sp)
                batch_out = [r.outputs[0].text for r in results]
                tier_usage[tier_num] += 1
                break
            except Exception as e:
                last_err = str(e)[:120]
                print(f"\n  Batch {i} tier {tier_num} failed: {last_err}")

        if batch_out is None:
            batch_out = [
                BATCH_ERROR_TEMPLATE.format(err=last_err.replace('"', "'"))
                for _ in batch_orig
            ]

        raw_outputs.extend(batch_out)
        batches_since_ckpt += 1

        # ---- Checkpoint save every CHECKPOINT_EVERY batches -----------------
        if batches_since_ckpt >= CHECKPOINT_EVERY:
            n_saved = save_checkpoint(
                raw_outputs, pending_indices, df, checkpoint_rows
            )
            print(f"\n  ✓ Checkpoint saved: {n_saved} total rows → {CHECKPOINT_FILE}  "
                  f"[batch {batch_num+1}/{(len(chat_messages)+BATCH_SIZE-1)//BATCH_SIZE}]")
            batches_since_ckpt = 0

    # Final checkpoint save (catches any remaining batches)
    n_saved = save_checkpoint(raw_outputs, pending_indices, df, checkpoint_rows)
    print(f"\n  ✓ Final checkpoint saved: {n_saved} total rows → {CHECKPOINT_FILE}")

    tier_summary = "  ".join(
        f"Tier{n}({['16k','8k','4k'][n-1]}): {tier_usage.get(n,0)}"
        for n in range(1, len(active_tiers)+1)
    )
    print(f"\nTier usage → {tier_summary}")

    # ---- Parse outputs — read final checkpoint for complete merged result -----
    # The checkpoint already contains both previously-done rows (from prior runs)
    # AND the rows we just finished. Load it as the definitive complete result.
    print(f"\nLoading final results from checkpoint: {CHECKPOINT_FILE}")
    ckpt_final = pd.read_csv(CHECKPOINT_FILE)
    print(f"Total rows in checkpoint: {len(ckpt_final):,}")

    # The checkpoint already has oare_id, original_id, source_pdf + parsed fields.
    # We just need to enforce the correct column order and hand off to postprocess.
    col_order = [
        'oare_id', 'original_id', 'source_pdf',
        'status',
        'original_text_found',
        'original_language_code',
        'english_translation',
        'confidence',
        'is_truncated',
        'reason'
    ]
    for c in col_order:
        if c not in ckpt_final.columns:
            ckpt_final[c] = None

    # Drop the internal _df_index tracking column
    merged = ckpt_final[col_order].copy()

    parse_errors = (merged['status'] == 'PARSE_ERROR').sum()
    print(f"Parse errors in checkpoint: {parse_errors} / {len(merged)}")

    # ---- Post-process -------------------------------------------------------
    final = postprocess_results(merged)

    # ---- Save ---------------------------------------------------------------
    final.to_csv(OUTPUT_CSV, index=False)

    # ---- Summary ------------------------------------------------------------
    valid_only = final[final['status'] == 'VALID']
    invalid_only = final[final['status'] != 'VALID']

    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Rows saved total:         {len(final):,}")
    print(f"VALID extractions:        {len(valid_only):,}")
    print(f"Unique oare_ids (valid):  {valid_only['oare_id'].nunique():,}")

    if len(valid_only) > 0:
        print(f"\nLanguage breakdown (VALID):")
        print(valid_only['original_language_code'].value_counts().to_string())
        print(f"\nConfidence breakdown (VALID):")
        print(valid_only['confidence'].value_counts().to_string())
        print(f"\nNon-EN translations that were translated to EN: "
              f"{len(valid_only[~valid_only['original_language_code'].isin(['EN','AK'])])}")

    if len(invalid_only) > 0:
        print(f"\nTop rejection reasons (INVALID):")
        reasons = invalid_only['reason'].fillna('no reason').str[:90]
        for r, c in Counter(reasons).most_common(10):
            print(f"  {c}x  {r}")

    # Sample preview
    print(f"\n--- Sample VALID EN extractions ---")
    sample_en = valid_only[valid_only['original_language_code'] == 'EN'].head(3)
    for _, r in sample_en.iterrows():
        print(f"  ID:  {r['original_id']}")
        print(f"  EN:  {str(r['english_translation'])[:200]}")
        print()

    sample_non_en = valid_only[
        ~valid_only['original_language_code'].isin(['EN', 'AK'])
    ].head(3)
    if len(sample_non_en) > 0:
        print(f"--- Sample non-EN (auto-translated to EN) ---")
        for _, r in sample_non_en.iterrows():
            print(f"  ID:    {r['original_id']}")
            print(f"  Lang:  {r['original_language_code']}")
            print(f"  Orig:  {str(r['original_text_found'])[:150]}")
            print(f"  EN:    {str(r['english_translation'])[:150]}")
            print()

    print(f"\nOutput saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

# %% [code] {"jupyter":{"outputs_hidden":false}}
