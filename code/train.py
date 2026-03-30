# %% [code] {"execution":{"iopub.status.busy":"2026-03-09T09:05:26.022347Z","iopub.execute_input":"2026-03-09T09:05:26.022704Z","iopub.status.idle":"2026-03-09T09:05:30.605434Z","shell.execute_reply.started":"2026-03-09T09:05:26.022687Z","shell.execute_reply":"2026-03-09T09:05:30.604699Z"}}
!pip install sacrebleu

# %% [code] {"execution":{"iopub.status.busy":"2026-03-09T09:05:30.606743Z","iopub.execute_input":"2026-03-09T09:05:30.606900Z","execution_failed":"2026-03-09T10:01:11.807Z"}}
"""
Deep Past — v9: Self-contained data engineering + training
====================================================================

KEY INSIGHT (why glossing was 0 in all prior runs):
  data_engineering_v3.py ran in a SEPARATE notebook that had competition
  data attached. The training notebook (v7/v8) did NOT have competition
  data → build_lexicon() returned {} silently → 0 glossed pairs.

  v9 FIXES THIS by reading everything directly from competition data
  in the same script. One notebook, one attachment.

WHAT v9 DOES (vs v8):
  1. Builds lexicon inline from competition CSVs (54,511 entries vs 0)
       - Fixes eBL_Dictionary column name: 'definition' (not 'english')
       - Accent-normalized secondary lookup for accented source words
  2. Builds document pairs from train.csv directly (not pre-built CSV)
  3. Rebuilds sentence pairs from Sentences_Oare_FirstWord_LinNum.csv
  4. Loads extracted AKT pairs from extract_train.csv
  5. Generates proper noun training pairs (1,500 pairs)
  6. Uses correct thresholds per pair type:
       document  → 0.15  (full tablets, many common words)
       sentence  → 0.08  (shorter, still OK)
       extracted → 0.03  (proper names dominate, any gloss is useful)
link:https://www.kaggle.com/code/sudharsananh/fork-of-vesudeep/edit/run/302359743

EXPECTED TRAINING DATA:
  document_orig         ~1,545
  document_glossed      ~1,545
  sentence_orig         ~  820
  sentence_glossed      ~  810
  extracted_akt         ~  211
  extracted_akt_glossed ~  175
  proper_noun           ~  900
  proper_noun_glossed   ~  600
  ─────────────────────────────
  Total                 ~6,600 pairs   (vs 4,937 in v8)
"""

import torch
import unicodedata
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from datasets import Dataset
import pandas as pd
import numpy as np
import os, gc, re, math, shutil
import sacrebleu

# ============================================================
#  PATHS
# ============================================================

# Competition data
COMP_DIR     = "/kaggle/input/datasets/sudharsananh/deep-past-data"
TRAIN_CSV    = f"{COMP_DIR}/train.csv"
SENT_CSV     = f"{COMP_DIR}/Sentences_Oare_FirstWord_LinNum.csv"
LEXICON_CSV  = f"{COMP_DIR}/OA_Lexicon_eBL.csv"
DICT_CSV     = f"{COMP_DIR}/eBL_Dictionary.csv"
PUB_TXT_CSV  = f"{COMP_DIR}/published_texts.csv"

# Extracted AKT pairs (upload extract_train.csv as dataset: sudharsananh/extractpub)
EXTRACT_FILE = "/kaggle/input/datasets/sudharsananh/extractpub/extract_train.csv"

# Base model
PUBLIC_CHECKPOINT = "/kaggle/input/datasets/assiaben/final-byt5/byt5-akkadian-optimized-34x"

# Output
OUTPUT_DIR  = "akkadian-byt5-model-v9"
FINAL_MODEL = f"{OUTPUT_DIR}/final"

# ============================================================
#  TRAINING HYPERPARAMETERS  (H100, same as v7/v8)
# ============================================================
EPOCHS       = 20
MAX_LENGTH   = 512
BATCH_SIZE   = 24
GRAD_ACCUM   = 2
LR           = 1e-5
WEIGHT_DECAY = 0.01
FP16         = False
BF16         = True
PATIENCE     = 5
SEED         = 42
GRAD_CKPT    = False
NUM_WORKERS  = 4

# ============================================================
#  CHARACTER MAPS
# ============================================================
H_MAP         = str.maketrans("ḫḪ", "hH")
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# ============================================================
#  EXTRACTED PAIR BLACKLIST  (known bad oare_ids)
# ============================================================
BAD_OARE_PREFIXES = {
    "c0aaa098", "2eb4b26f", "3438632c", "51293c64", "8376cbda",
    "89179664", "d52b7a5b", "da13d2c6", "e70d1d21", "f98df4c2",
}

COMMENTARY_RE = re.compile(
    r"^(Note concerning|Account of|Transport contract\.|Sealed by|"
    r"Empty envelope|Tablet concerning|No valid English translation|"
    r"See no\.|Cf\. no\.|List of|Memo concerning)",
    re.IGNORECASE,
)

# ============================================================
#  OCR ARTIFACT FIXER
# ============================================================
OCR_FIXES = [
    (r"A\$\$ur",      "Aššur"),  (r"A\$\$Ur",     "Aššur"),
    (r"Pu\$ur",       "Puzur"),  (r"I\$tar",       "Ištar"),
    (r"I\$TAR",       "IŠTAR"),  (r"\$alim",        "Šalim"),
    (r"\$u-",         "Šu-"),    (r"\$U-",          "ŠU-"),
    (r"\$a-",         "ša-"),    (r"\$i-",          "ši-"),
    (r"\$e-",         "še-"),    (r"(\w)\$\$(\w)",  r"\1šš\2"),
    (r"(\w)\$(\w)",   r"\1š\2"), (r"^\$",           "š"),
    (r"§",            "Š"),      (r"ª",             "ā"),
    (r"‰",            "ī"),      (r"£",             "ū"),
]

def fix_ocr(text: str) -> str:
    if not isinstance(text, str): return text
    for pat, rep in OCR_FIXES:
        text = re.sub(pat, rep, text)
    return text


# ============================================================
#  ACCENT NORMALIZATION
# ============================================================
def _strip_accents(s: str) -> str:
    """NFD decompose + strip combining marks.
    'qí-bi4-ma'→'qi-bi4-ma', 'ù'→'u', 'ša'→'sa', 'áb'→'ab'"""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# ============================================================
#  PREPROCESSING
# ============================================================
def _dedup_gap(text: str) -> str:
    return re.sub(r"(?:<gap>\s*){2,}", "<gap> ", text).strip()


def preprocess_source(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = text.translate(H_MAP).translate(SUBSCRIPT_MAP)
    text = re.sub(r"(\.\s*\.+|…|\[\s*\.\s*\.+\s*\]|\[…\])", " <gap> ", text)
    text = re.sub(r"\b(xx+)\b", " <gap> ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)x(?!\w)", " <gap> ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"[!?/]", "", text)
    text = re.sub(r"[⌈⌋⌊]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _dedup_gap(text)


def preprocess_target(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = re.sub(r"'s\b", "s", text)
    text = re.sub(r"'\b", "", text)
    text = re.sub(r'[""„"«»]', "", text)
    text = re.sub(r"(?<!\w)'(?!\w)", "", text)
    text = re.sub(r"(\.\s*\.+|…)", " <gap> ", text)
    text = re.sub(r"(?<!\w)x(?!\w)", " <gap> ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return _dedup_gap(text)


def is_valid_pair(src: str, tgt: str) -> bool:
    if not src or not tgt: return False
    sw, tw = src.split(), tgt.split()
    if len(tw) < 3 or len(sw) < 2: return False
    if len(sw) > 10 and len(tw) < 8 and len(sw) / max(len(tw), 1) > 4.0: return False
    clean = tgt.replace("<gap>", "").strip()
    return len(clean.split()) >= 3


# ============================================================
#  LEXICON BUILDER
#  Reads directly from competition data (OA_Lexicon_eBL.csv + eBL_Dictionary.csv)
#
#  Critical bug fix vs v7/v8:
#    eBL_Dictionary.csv columns = ['word', 'definition', 'derived_from']
#    Previous code searched for 'english'/'English' which don't exist → fell back
#    to lexeme-only (still 54k entries, but no English meaning strings)
#    Fix: search for 'definition' first
#
#  Returns: (lexicon, lexicon_norm)
#    lexicon      — exact key (lowercased form/norm) → gloss string
#    lexicon_norm — accent-stripped key              → gloss string (fallback)
# ============================================================
def build_lexicon():
    print("Building lexicon...")
    print(f"  OA_Lexicon_eBL.csv : {'OK' if os.path.exists(LEXICON_CSV) else 'MISSING'}")
    print(f"  eBL_Dictionary.csv : {'OK' if os.path.exists(DICT_CSV)    else 'MISSING'}")

    lexicon = {}

    if not os.path.exists(LEXICON_CSV):
        print("  WARNING: Lexicon CSV not found — attach competition data to this notebook!")
        return {}, {}

    # --- Step 1: OA_Lexicon_eBL.csv → form/norm → lexeme ---
    lex_df = pd.read_csv(LEXICON_CSV)
    form_to_lex = {}
    for _, row in lex_df.iterrows():
        if pd.isna(row.get("form")) or pd.isna(row.get("lexeme")): continue
        form   = str(row["form"]).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
        lexeme = str(row["lexeme"]).strip()
        form_to_lex[form] = lexeme
        if pd.notna(row.get("norm")):
            norm = str(row["norm"]).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
            if norm not in form_to_lex:
                form_to_lex[norm] = lexeme
    print(f"  form→lexeme mappings: {len(form_to_lex):,}")

    # --- Step 2: eBL_Dictionary.csv → lexeme → English gloss ---
    # IMPORTANT: columns are 'word', 'definition', 'derived_from'
    # (NOT 'lemma'/'english' as previous code assumed)
    lex_to_eng = {}
    if os.path.exists(DICT_CSV):
        dict_df = pd.read_csv(DICT_CSV)
        print(f"  eBL_Dictionary columns: {dict_df.columns.tolist()}")
        # Search for English column — 'definition' is the real name
        eng_col = next(
            (c for c in ["definition", "Definition", "english", "English",
                          "meaning", "Meaning", "gloss", "translation"]
             if c in dict_df.columns), None)
        lem_col = next(
            (c for c in ["word", "Word", "lemma", "Lemma", "headword"]
             if c in dict_df.columns), None)
        if eng_col and lem_col:
            for _, row in dict_df.iterrows():
                if pd.isna(row.get(lem_col)) or pd.isna(row.get(eng_col)): continue
                lemma = str(row[lem_col]).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
                gloss = str(row[eng_col]).split(";")[0].split(",")[0].strip()[:30]
                if gloss and len(gloss) > 1 and not gloss.startswith("?"):
                    lex_to_eng[lemma] = gloss
            print(f"  lemma→English: {len(lex_to_eng):,} mappings (eng_col='{eng_col}')")
        else:
            print(f"  WARNING: eng_col={eng_col} lem_col={lem_col} — using lexeme as gloss")

    # --- Step 3: merge: form → English (or lexeme fallback) ---
    for form, lexeme in form_to_lex.items():
        lexicon[form] = lex_to_eng.get(lexeme.lower(), lexeme)

    # --- Step 4: accent-normalized secondary lookup ---
    lexicon_norm = {}
    for k, v in lexicon.items():
        nk = _strip_accents(k)
        if nk not in lexicon_norm:
            lexicon_norm[nk] = v

    print(f"  Lexicon TOTAL: {len(lexicon):,} entries (exact) | "
          f"{len(lexicon_norm):,} (accent-norm)")
    return lexicon, lexicon_norm


# ============================================================
#  GLOSS INJECTION
#  min_match_ratio controls how many unique source words must hit
#  the lexicon before we actually inject:
#    0.15 → document pairs  (full tablets, ~30% common words)
#    0.08 → sentence pairs  (shorter, fewer function words)
#    0.03 → extracted pairs (heavy in proper names/sumerograms)
# ============================================================
def inject_glosses(
    text: str,
    lexicon: dict,
    lexicon_norm: dict = None,
    max_glosses: int = None,
    min_match_ratio: float = 0.15,
) -> str:
    if not text or not lexicon: return text
    words = text.split()
    if max_glosses is None:
        max_glosses = min(12, max(3, math.ceil(len(words) * 0.6)))

    found, seen = [], set()
    for word in words:
        clean = re.sub(r"[.,;:!?]$", "", word.strip().lower())
        gloss = None

        # 1. exact match
        if clean in lexicon and clean not in seen:
            raw = lexicon[clean]
            gloss = re.sub(r"[<>{}]", "", raw).strip()[:25]
            if not gloss or len(gloss) < 2: gloss = None

        # 2. accent-stripped fallback
        if gloss is None and lexicon_norm:
            cn = _strip_accents(clean)
            if cn in lexicon_norm and cn not in seen:
                raw = lexicon_norm[cn]
                gloss = re.sub(r"[<>{}]", "", raw).strip()[:25]
                if not gloss or len(gloss) < 2: gloss = None
                else: clean = cn

        if gloss and clean not in seen:
            found.append(f"{word}={gloss}")
            seen.add(clean)

    if not found or len(seen) / max(len(words), 1) < min_match_ratio:
        return text
    if len(found) > max_glosses:
        found = found[:max_glosses]
    return "[" + "; ".join(found) + "] " + text


# ============================================================
#  DATA SOURCE 1: DOCUMENT PAIRS  (from train.csv)
#  1,545 document-level Akkadian→English pairs + glossed variants
# ============================================================
def build_document_pairs(lexicon: dict, lexicon_norm: dict) -> pd.DataFrame:
    print("\n[DATA] Document pairs from train.csv...")
    if not os.path.exists(TRAIN_CSV):
        print("  MISSING: train.csv — check competition data attachment")
        return pd.DataFrame(columns=["source", "target", "type", "oare_id"])

    train_df = pd.read_csv(TRAIN_CSV)
    print(f"  Raw rows: {len(train_df)}")

    rows = []
    skipped = 0
    for _, row in train_df.iterrows():
        src = preprocess_source(str(row.get("transliteration", "")))
        tgt = preprocess_target(str(row.get("translation", "")))
        if not is_valid_pair(src, tgt):
            skipped += 1
            continue
        oid = str(row.get("oare_id", ""))
        rows.append({"source": src, "target": tgt, "type": "document_orig", "oare_id": oid})
        glossed = inject_glosses(src, lexicon, lexicon_norm, min_match_ratio=0.15)
        if glossed != src:
            rows.append({"source": glossed, "target": tgt,
                         "type": "document_glossed", "oare_id": oid})

    df = pd.DataFrame(rows)
    orig_n    = (df["type"] == "document_orig").sum()
    glossed_n = (df["type"] == "document_glossed").sum()
    print(f"  document_orig:    {orig_n}  |  document_glossed: {glossed_n}  "
          f"|  skipped: {skipped}")
    return df


# ============================================================
#  DATA SOURCE 2: SENTENCE PAIRS  (from Sentences_Oare_FirstWord_LinNum.csv)
#  Extracts sentence-span pairs aligned from full tablets
# ============================================================
def build_sentence_pairs(lexicon: dict, lexicon_norm: dict,
                          doc_oare_ids: set) -> pd.DataFrame:
    print("\n[DATA] Sentence pairs from Sentences_Oare_FirstWord_LinNum.csv...")
    if not os.path.exists(SENT_CSV) or not os.path.exists(TRAIN_CSV):
        print("  MISSING sentence CSV or train.csv")
        return pd.DataFrame(columns=["source", "target", "type"])

    sent_df  = pd.read_csv(SENT_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"  Sentence CSV rows: {len(sent_df)}  columns: {sent_df.columns.tolist()}")

    # Find the UUID/ID column in sentence CSV
    id_col = next((c for c in ["text_uuid", "oare_id", "text_id", "uuid"]
                   if c in sent_df.columns), None)
    fw_col = next((c for c in ["first_word_number", "word_number", "word_num"]
                   if c in sent_df.columns), None)
    tr_col = next((c for c in ["translation", "sentence_translation", "english"]
                   if c in sent_df.columns), None)

    if not id_col or not fw_col:
        print(f"  Cannot find required columns. Available: {sent_df.columns.tolist()}")
        return pd.DataFrame(columns=["source", "target", "type"])

    print(f"  Using: id_col='{id_col}' fw_col='{fw_col}' tr_col='{tr_col}'")

    train_idx = train_df.set_index("oare_id")
    overlap   = set(train_idx.index) & set(sent_df[id_col].dropna())
    print(f"  Overlapping documents: {len(overlap)}")

    rows = []
    for doc_id, group in sent_df.groupby(id_col):
        if doc_id not in overlap: continue
        full_words = str(train_idx.loc[doc_id, "transliteration"]).split()
        sentences  = group.sort_values(fw_col).reset_index(drop=True)

        for i, row in sentences.iterrows():
            tgt_raw = str(row.get(tr_col, "")) if tr_col else ""
            if not tgt_raw or tgt_raw == "nan": continue
            start = max(0, int(row[fw_col]) - 1)
            end   = int(sentences.loc[i + 1, fw_col]) - 1 \
                    if i < len(sentences) - 1 else len(full_words)
            span  = " ".join(full_words[start:end]).strip()
            if len(span) < 3: continue

            src = preprocess_source(span)
            tgt = preprocess_target(tgt_raw)
            if not is_valid_pair(src, tgt): continue

            rows.append({"source": src, "target": tgt, "type": "sentence_orig"})
            glossed = inject_glosses(src, lexicon, lexicon_norm, min_match_ratio=0.08)
            if glossed != src:
                rows.append({"source": glossed, "target": tgt, "type": "sentence_glossed"})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source","target","type"])
    if len(df):
        print(f"  sentence_orig: {(df['type']=='sentence_orig').sum()}  |  "
              f"sentence_glossed: {(df['type']=='sentence_glossed').sum()}")
    return df


# ============================================================
#  DATA SOURCE 3: EXTRACTED AKT PAIRS  (from extract_train.csv)
#  211 pairs extracted by LLM agent from AKT publication PDFs
# ============================================================
def build_extracted_pairs(lexicon: dict, lexicon_norm: dict) -> pd.DataFrame:
    """
    Load and clean extracted AKT publication translations.

    NO source-text dedup vs doc pairs. Rationale:
      Extracted pairs come from the same tablets as train.csv BUT are
      translated by different scholars from AKT publication series.
      Same Akkadian source + different expert English target = valuable:
      the model learns that translation is not one-to-one.
      We only remove: bad oare_ids, commentary, and duplicate targets
      within the extracted set itself.
    """
    print(f"\n[DATA] Extracted AKT pairs from {EXTRACT_FILE}...")
    if not os.path.exists(EXTRACT_FILE):
        print(f"  MISSING: {EXTRACT_FILE}")
        return pd.DataFrame(columns=["source", "target", "type"])

    ext_df = pd.read_csv(EXTRACT_FILE)
    print(f"  Raw pairs          : {len(ext_df)}")

    # 1. Blacklisted oare_ids
    ext_df = ext_df[
        ~ext_df["oare_id"].apply(
            lambda x: any(str(x).startswith(p) for p in BAD_OARE_PREFIXES)
        )
    ]
    print(f"  After blacklist    : {len(ext_df)}")

    # 2. Commentary/metadata targets
    ext_df = ext_df[~ext_df["target"].astype(str).str.match(COMMENTARY_RE)]
    print(f"  After commentary   : {len(ext_df)}")

    # 3. Deduplicate within extracted set by target text only
    ext_df = ext_df.drop_duplicates(subset=["target"])
    print(f"  After target dedup : {len(ext_df)}")

    if len(ext_df) == 0:
        return pd.DataFrame(columns=["source", "target", "type"])

    # 4. Preprocessing
    ext_df = ext_df.copy()
    ext_df["target"] = ext_df["target"].apply(fix_ocr)
    ext_df["source"] = ext_df["source"].apply(preprocess_source)
    ext_df["target"] = ext_df["target"].apply(preprocess_target)
    ext_df = ext_df.dropna(subset=["source", "target"])
    ext_df = ext_df[ext_df["source"].str.strip().ne("") & ext_df["target"].str.strip().ne("")]
    ext_df["_wc"] = ext_df["target"].str.split().str.len()
    ext_df = ext_df[ext_df["_wc"] >= 5].drop(columns=["_wc"])
    print(f"  After preprocessing: {len(ext_df)} clean pairs")

    # 5. Build base + glossed rows
    base_rows = [{"source": r["source"], "target": r["target"],
                  "type": "extracted_akt"} for _, r in ext_df.iterrows()]
    glossed_rows = []
    for _, row in ext_df.iterrows():
        g = inject_glosses(row["source"], lexicon, lexicon_norm, min_match_ratio=0.03)
        if g != row["source"]:
            glossed_rows.append({"source": g, "target": row["target"],
                                  "type": "extracted_akt_glossed"})

    df = pd.DataFrame(base_rows + glossed_rows)
    print(f"  extracted_akt:         {len(base_rows)}")
    print(f"  extracted_akt_glossed: {len(glossed_rows)}")
    if "confidence" in ext_df.columns:
        print(f"  Confidence: {ext_df['confidence'].value_counts().to_dict()}")
    wc = ext_df["target"].str.split().str.len()
    print(f"  Target words — min:{wc.min()} median:{wc.median():.0f} "
          f"mean:{wc.mean():.0f} max:{wc.max()}")
    return df


# ============================================================
#  DATA SOURCE 4: PROPER NOUN PAIRS  (from OA_Lexicon_eBL.csv)
#  Small set of template pairs to teach name transliteration.
#  IMPORTANT: Keep this SMALL (200-300 max). These are synthetic
#  templates — too many overwhelms real training signal.
#  In v9 test: 3000 PN pairs (1500 + 1500 glossed) out of 7689
#  total = 39% synthetic → caused score to drop from 36→32.
# ============================================================
def build_proper_noun_pairs(lexicon: dict, lexicon_norm: dict,
                             max_pairs: int = 200) -> pd.DataFrame:
    print(f"\n[DATA] Proper noun pairs (max {max_pairs} — kept small, synthetic)...")
    if not os.path.exists(LEXICON_CSV):
        return pd.DataFrame(columns=["source", "target", "type"])

    lex_df  = pd.read_csv(LEXICON_CSV)
    pn_rows = lex_df[lex_df["type"].isin(["PN", "GN"])].dropna(subset=["form", "norm"])

    # Only use templates that match real test patterns (address formulas)
    templates = [
        ("a-na {form} qi-bi-ma",   "Speak to {norm}:"),
        ("um-ma {form}-ma",         "Thus {norm}:"),
        ("IGI {form}",              "Witness: {norm}"),
        ("{form} DUMU {form2}",     "{norm}, son of {norm2}"),
    ]

    pn_pairs, seen_forms = [], set()
    for _, row in pn_rows.iterrows():
        form = str(row["form"]).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
        norm = str(row["norm"]).strip()
        if form not in seen_forms and len(form) > 2 and len(norm) > 1:
            pn_pairs.append((form, norm))
            seen_forms.add(form)

    np.random.seed(SEED)
    np.random.shuffle(pn_pairs)

    rows = []
    n_templates = len(templates)
    for i, (form, norm) in enumerate(pn_pairs[:max_pairs]):
        tmpl_idx = i % n_templates
        src_t, tgt_t = templates[tmpl_idx]
        j = (i + 1) % len(pn_pairs)
        form2, norm2 = pn_pairs[j]
        src = src_t.format(form=form, form2=form2)
        tgt = tgt_t.format(norm=norm, norm2=norm2)
        if len(src.split()) < 2 or len(tgt.split()) < 2: continue
        rows.append({"source": src, "target": tgt, "type": "proper_noun"})
        # No glossed variants for PN pairs — adds noise without signal

    df = pd.DataFrame(rows)
    print(f"  proper_noun: {len(df)}")
    return df


# ============================================================
#  MASTER DATA LOADER
# ============================================================
def load_all_data():
    print("=" * 60)
    print("Deep Past — v9 Data Engineering")
    print("=" * 60)

    # 1. Lexicon (reads from competition data — must be attached)
    lexicon, lexicon_norm = build_lexicon()
    if not lexicon:
        raise RuntimeError(
            "Lexicon is empty! Attach the competition dataset to this notebook.\n"
            f"Expected at: {COMP_DIR}"
        )

    # 2. Document pairs (train.csv)
    doc_df = build_document_pairs(lexicon, lexicon_norm)

    # 3. Sentence pairs (Sentences_Oare_FirstWord_LinNum.csv)
    sent_df = build_sentence_pairs(lexicon, lexicon_norm, set())

    # 4. Extracted AKT pairs — NO source-text dedup vs doc pairs.
    #    These are alternate AKT-series translations of the same tablets.
    #    Same Akkadian + different expert English = useful training signal.
    ext_df = build_extracted_pairs(lexicon, lexicon_norm)

    # 5. Proper noun pairs — small (200), no glossed variants
    pn_df = build_proper_noun_pairs(lexicon, lexicon_norm, max_pairs=200)

    # --- Combine ---
    all_parts = [p for p in [doc_df, sent_df, ext_df, pn_df] if len(p) > 0]
    full_df = pd.concat(all_parts, ignore_index=True)

    # Drop only exact (source, target) duplicates — keep same-source/different-target
    before = len(full_df)
    full_df.drop_duplicates(subset=["source", "target"], inplace=True)
    print(f"\n[DEDUP] {before} → {len(full_df)} pairs after (source,target) dedup")

    # --- Val split: sentence pairs only, 10% (matches v7/v8 baseline) ---
    sent_mask  = full_df["type"].isin(["sentence_orig", "sentence_glossed"])
    sent_pool  = full_df[sent_mask].copy()
    non_sent   = full_df[~sent_mask].copy()

    val_n      = max(50, int(len(sent_pool) * 0.10))
    val_df     = sent_pool.sample(n=min(val_n, len(sent_pool)), random_state=SEED)
    train_sent = sent_pool.drop(val_df.index)

    train_df = pd.concat([non_sent, train_sent], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nFinal split:")
    print(f"  Train: {len(train_df)} pairs")
    print(f"  Val  : {len(val_df)} pairs")
    print(f"  Train types:\n{train_df['type'].value_counts().to_string()}")

    return train_df, val_df, lexicon


# ============================================================
#  METRICS
# ============================================================
def postprocess_eval(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.translate(H_MAP).translate(SUBSCRIPT_MAP)
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _dedup_gap(text)


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds  = np.clip(preds,  0, len(tokenizer) - 1)
        labels = np.clip(labels, 0, len(tokenizer) - 1)
        dec_preds  = [postprocess_eval(p) for p in
                      tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        dec_labels = [l.strip() for l in
                      tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu     = sacrebleu.corpus_bleu(dec_preds, [dec_labels])
        chrf     = sacrebleu.corpus_chrf(dec_preds, [dec_labels], word_order=2)
        combined = (math.sqrt(bleu.score * chrf.score)
                    if bleu.score > 0 and chrf.score > 0 else 0.0)
        return {"bleu": round(bleu.score, 2),
                "chrf": round(chrf.score, 2),
                "combined": round(combined, 4)}
    return compute_metrics


# ============================================================
#  TOKENIZATION
# ============================================================
def tokenize_df(df: pd.DataFrame, tokenizer):
    dataset = Dataset.from_pandas(df[["source", "target"]].reset_index(drop=True))
    def preprocess(examples):
        inputs  = ["translate Akkadian to English: " + str(s) for s in examples["source"]]
        targets = [str(t) for t in examples["target"]]
        return tokenizer(inputs, text_target=targets,
                         max_length=MAX_LENGTH, truncation=True, padding=False)
    return dataset.map(preprocess, batched=True,
                       remove_columns=dataset.column_names,
                       num_proc=NUM_WORKERS)


# ============================================================
#  TRAINING
# ============================================================
def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def train(train_df: pd.DataFrame, val_df: pd.DataFrame,
          base_model: str) -> str:
    cleanup_gpu()
    print(f"\n{'='*60}")
    print("TRAINING v9 — inline data engineering + 54k lexicon + soup")
    print(f"Base: {base_model}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tok_train = tokenize_df(train_df, tokenizer)
    tok_val   = tokenize_df(val_df,   tokenizer)
    print(f"Train: {len(tok_train)} | Val: {len(tok_val)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    print(f"Parameters: {model.num_parameters():,}")

    steps_per_epoch = math.ceil(len(tok_train) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps     = steps_per_epoch * EPOCHS
    print(f"Steps/epoch: {steps_per_epoch}  Total: {total_steps}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        fp16=FP16, bf16=BF16,
        gradient_checkpointing=GRAD_CKPT,
        optim="adafactor",
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        eval_strategy="steps",  eval_steps=100,
        save_strategy="steps",  save_steps=100,
        logging_steps=25,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        metric_for_best_model="combined",
        greater_is_better=True,
        load_best_model_at_end=True,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        push_to_hub=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model,
            label_pad_token_id=-100, pad_to_multiple_of=8),
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    os.makedirs(FINAL_MODEL, exist_ok=True)
    trainer.save_model(FINAL_MODEL)
    tokenizer.save_pretrained(FINAL_MODEL)
    print(f"\n✓ Saved: {FINAL_MODEL}")

    result = trainer.evaluate()
    print(f"\nVal (sentence-level):")
    print(f"  BLEU={result.get('eval_bleu')}  "
          f"chrF={result.get('eval_chrf')}  "
          f"Combined={result.get('eval_combined')}")

    for item in os.listdir(OUTPUT_DIR):
        if item.startswith("checkpoint"):
            shutil.rmtree(os.path.join(OUTPUT_DIR, item), ignore_errors=True)

    del model, trainer, tokenizer
    cleanup_gpu()
    return FINAL_MODEL


# ============================================================
#  MODEL SOUP  alpha=0.7 (unchanged)
# ============================================================
def model_soup(path_a: str, path_b: str, out: str, alpha: float = 0.7):
    print(f"\nModel soup: {alpha*100:.0f}% fine-tuned + {(1-alpha)*100:.0f}% public")
    tok = AutoTokenizer.from_pretrained(path_a)
    ma  = AutoModelForSeq2SeqLM.from_pretrained(path_a)
    mb  = AutoModelForSeq2SeqLM.from_pretrained(path_b)
    sa, sb = ma.state_dict(), mb.state_dict()
    merged = {}
    for k in sa:
        if k in sb:
            w = alpha * sa[k].float() + (1 - alpha) * sb[k].float()
            merged[k] = w.to(sa[k].dtype)
        else:
            merged[k] = sa[k]
    ma.load_state_dict(merged)
    os.makedirs(out, exist_ok=True)
    ma.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"✓ Soup saved: {out}")
    del ma, mb
    cleanup_gpu()


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Deep Past — v9")
    print("=" * 60)

    # Sanity checks
    for path, label in [
        (PUBLIC_CHECKPOINT, "Base model"),
        (COMP_DIR,          "Competition data  (sudharsananh/deep-past-data)"),
        (EXTRACT_FILE,      "Extracted AKT pairs"),
    ]:
        exists = os.path.exists(path)
        print(f"  {'OK' if exists else 'MISSING'}: {label}  ({path})")
        if not exists and label != "Extracted AKT pairs":
            raise FileNotFoundError(f"Required: {path}")

    train_df, val_df, _ = load_all_data()
    final_path = train(train_df, val_df, base_model=PUBLIC_CHECKPOINT)

    soup_path = f"{OUTPUT_DIR}/soup"
    model_soup(final_path, PUBLIC_CHECKPOINT, soup_path, alpha=0.7)

    print(f"""
{'='*60}
DONE — v9

Inference: use soup_path = "{soup_path}"

WHAT CHANGED vs v8:
  ✓ Lexicon now actually loads (54,511 entries from competition CSV)
  ✓ eBL_Dictionary column fix: 'definition' (was searching 'english')
  ✓ Accent-normalized fallback lookup for accented source words
  ✓ Per-type thresholds: doc=0.15  sent=0.08  extracted=0.03
  ✓ Proper noun pairs: 1,500 (was 0 because lexicon was empty)
  ✓ No dependency on pre-built train_final.csv
  ✓ ~6,600 total training pairs (was 4,937)
{'='*60}
""")

# %% [code] {"execution":{"execution_failed":"2026-03-09T10:01:11.807Z"}}
# import json, os

# dataset_dir="/kaggle/working/akkadian-byt5-model-v5"

# meta={
#     "title":"akkadian-byt5-model-v4",
#     "id":"sudharsananh/akkadian-byt5-model-v5",
#     "licenses":[{"name":"CC0-1.0"}]
# }

# with open(os.path.join(dataset_dir,"dataset-metadata.json"),"w") as f:
#     json.dump(meta,f)

# from kaggle_secrets import UserSecretsClient
# import os

# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("sudharsananh")

# os.makedirs("/root/.kaggle", exist_ok=True)

# with open("/root/.kaggle/kaggle.json","w") as f:
#     f.write(secret_value_0)

# os.chmod("/root/.kaggle/kaggle.json",600)
# !kaggle datasets create -p /kaggle/working/akkadian-byt5-model-v5 --dir-mode tar


# %% [code]


# %% [code]
