# %% [code] {"execution":{"iopub.status.busy":"2026-03-10T17:47:32.297965Z","iopub.execute_input":"2026-03-10T17:47:32.299149Z","iopub.status.idle":"2026-03-10T17:47:37.257468Z","shell.execute_reply.started":"2026-03-10T17:47:32.299115Z","shell.execute_reply":"2026-03-10T17:47:37.256622Z"}}
# !pip install sacrebleu

# %% [code] {"execution":{"iopub.status.busy":"2026-03-10T17:52:09.546597Z","iopub.execute_input":"2026-03-10T17:52:09.547513Z","iopub.status.idle":"2026-03-10T17:53:11.752575Z","shell.execute_reply.started":"2026-03-10T17:52:09.547482Z","shell.execute_reply":"2026-03-10T17:53:11.751994Z"}}
"""
Deep Past — Inference v9 Final
=================================
Fixes applied:
  1. Tokenizer always from PUBLIC_MODEL (avoids corrupted extra_special_tokens)
  2. embed_tokens tied to shared.weight after load (safety net for any upload)
  3. No fraction conversion — test labels keep 0.3333 as floats
  4. Parentheses PRESERVED — appear in test labels
  5. Curly quotes → straight quotes (not deleted)
  6. ḫ/Ḫ → h/H in postprocessing output
  7. <gap> sentinel protected during char removal
  8. MIN_COVERAGE=0.06 for sentence-level test data
link:https://www.kaggle.com/code/ragunathravi/fork-of-big-gap-fix-inference/edit/run/302454888
"""

import os, gc, re, math, warnings
from contextlib import nullcontext
from typing import List, Dict

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import sacrebleu
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

# ============================================================
#  PATHS — update SOUP_MODEL after uploading new trained model
# ============================================================
SOUP_MODEL      = '/kaggle/input/models/sanjaynn/byt5-mbr-v4/pytorch/default/2/akkadian-byt5-model-v8/soup'
FINETUNED_MODEL = '/kaggle/input/models/sanjaynn/byt5-mbr-v4/pytorch/default/2/akkadian-byt5-model-v8/final'
PUBLIC_MODEL    = '/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1'

TEST_FILE   = '/kaggle/input/deep-past-initiative-machine-translation/test.csv'
LEXICON_CSV = '/kaggle/input/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv'
DICT_CSV    = '/kaggle/input/deep-past-initiative-machine-translation/eBL_Dictionary.csv'
SUBMISSION  = 'submission.csv'

# ============================================================
#  SETTINGS
# ============================================================
NUM_BEAMS          = 12
NUM_BEAMS_SHORT    = 10
NUM_SAMPLES        = 3
LENGTH_PENALTY     = 1.6
REPETITION_PENALTY = 1.2
MAX_LENGTH         = 512
MIN_COVERAGE       = 0.06   # sentence-level test data needs lower threshold
MBR_POOL_CAP       = 32
USE_MBR            = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

H_MAP         = str.maketrans('ḫḪ', 'hH')
SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')

print(f'Device : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')


# ============================================================
#  BF16
# ============================================================

def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    except Exception:
        return False

USE_BF16 = _bf16_supported()
print(f'BF16   : {USE_BF16}')

def _autocast_ctx():
    if USE_BF16:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


# ============================================================
#  MODEL LOADING
#  New v8 soup saves embed_tokens explicitly (physical clone fix).
#  We still re-tie as a safety net in case an older soup is loaded.
# ============================================================

def _fix_embeddings(model):
    """
    Safety net: if embed_tokens were not saved as separate keys
    (old soup without physical clone fix), tie them to shared.weight.
    For the new v8 soup this is a no-op since all three are already correct.
    """
    shared_ptr = model.shared.weight.data_ptr()
    enc_ptr    = model.encoder.embed_tokens.weight.data_ptr()
    dec_ptr    = model.decoder.embed_tokens.weight.data_ptr()

    if enc_ptr != shared_ptr or dec_ptr != shared_ptr:
        # embed_tokens were randomly initialized (old checkpoint without fix)
        # tie them back to shared.weight
        model.encoder.embed_tokens.weight = model.shared.weight
        model.decoder.embed_tokens.weight = model.shared.weight
        print('  ⚠  embed_tokens re-tied to shared.weight (old checkpoint)')
    else:
        print('  ✓ embed_tokens correctly saved as separate tensors')
    return model


def _validate_output(model, tokenizer) -> bool:
    """Quick sanity check — returns True if model produces coherent English."""
    enc = tokenizer(
        'translate Akkadian to English: a-na be-li-ia qi-bi-ma',
        return_tensors='pt'
    ).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(enc.input_ids, num_beams=4, max_new_tokens=60)
    sample = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f'  Sanity check output: "{sample[:80]}"')
    printable = sum(1 for c in sample if c.isprintable() and ord(c) < 512)
    return len(sample.strip()) > 3 and printable / max(len(sample), 1) > 0.80


def load_model():
    """
    Priority order:
      1. SOUP_MODEL   — new v8 trained model (best)
      2. FINETUNED_MODEL — final without soup
      3. PUBLIC_MODEL — fallback (35.9 LB baseline)
    """
    # Always tokenizer from PUBLIC — soup tokenizer_config may have
    # extra_special_tokens as list not dict, crashing newer transformers
    tokenizer = AutoTokenizer.from_pretrained(PUBLIC_MODEL)

    candidates = [
        (SOUP_MODEL,      'v8-soup-70/30',  True),
        (FINETUNED_MODEL, 'v8-final',       True),
        (PUBLIC_MODEL,    'public-baseline', False),
    ]

    for path, label, use_glosses in candidates:
        if not os.path.exists(path):
            print(f'  — {label}: not found, skipping')
            continue
        print(f'\n  Trying {label}...')
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE).eval()
            model = _fix_embeddings(model)

            if _validate_output(model, tokenizer):
                print(f'✓ Model : {label}')
                return model, tokenizer, use_glosses
            else:
                print(f'  ✗ {label} produces garbage — skipping')
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'  ✗ {label} failed: {e}')
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

    raise RuntimeError('All model candidates failed — check paths')


# ============================================================
#  LEXICON
# ============================================================

def build_lexicon() -> Dict[str, str]:
    print('\nBuilding lexicon...')
    lexicon: Dict[str, str] = {}
    if not os.path.exists(LEXICON_CSV):
        return lexicon

    # OA_Lexicon_eBL.csv: type, form, norm, lexeme, eBL, ...
    lex_df = pd.read_csv(LEXICON_CSV)
    form_to_lex: Dict[str, str] = {}
    for _, row in lex_df.iterrows():
        if pd.isna(row.get('form')) or pd.isna(row.get('lexeme')):
            continue
        form   = str(row['form']).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
        lexeme = str(row['lexeme']).strip()
        form_to_lex[form] = lexeme
        if pd.notna(row.get('norm')):
            norm = str(row['norm']).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
            if norm not in form_to_lex:
                form_to_lex[norm] = lexeme

    if not os.path.exists(DICT_CSV):
        print(f'  Lexicon (lexeme only): {len(form_to_lex):,} entries')
        return dict(form_to_lex)

    # eBL_Dictionary.csv: word, definition, derived_from
    dict_df = pd.read_csv(DICT_CSV)
    print(f'  eBL_Dictionary columns: {dict_df.columns.tolist()}')

    eng_col = next((c for c in ['definition', 'Definition', 'english', 'English',
                                  'meaning', 'Meaning', 'gloss', 'translation']
                    if c in dict_df.columns), None)
    lem_col = next((c for c in ['word', 'Word', 'lemma', 'Lemma', 'headword']
                    if c in dict_df.columns), None)
    print(f'  eng_col={eng_col}  lem_col={lem_col}')

    if eng_col and lem_col:
        lex_to_eng: Dict[str, str] = {}
        for _, row in dict_df.iterrows():
            if pd.isna(row.get(lem_col)) or pd.isna(row.get(eng_col)):
                continue
            lemma = str(row[lem_col]).translate(H_MAP).translate(SUBSCRIPT_MAP).strip().lower()
            gloss = str(row[eng_col]).split(';')[0].split(',')[0].strip()[:30]
            if gloss and len(gloss) > 1 and not gloss.startswith('?'):
                lex_to_eng[lemma] = gloss
        for form, lex in form_to_lex.items():
            lexicon[form] = lex_to_eng.get(lex.lower(), lex)
        print(f'  lex_to_eng: {len(lex_to_eng):,}  total: {len(lexicon):,} entries')
    else:
        lexicon = dict(form_to_lex)
        print(f'  WARNING: eng_col not found — lexeme used as gloss')
        print(f'  Lexicon: {len(lexicon):,} entries')

    return lexicon


# ============================================================
#  PREPROCESSING
# ============================================================

_GAP_UNIFIED_RE = re.compile(
    r'<\s*big[\s_\-]*gap\s*>'
    r'|<\s*gap\s*>'
    r'|\bbig[\s_\-]*gap\b'
    r'|\bx(?:\s+x)+\b'
    r'|\.{3,}|…+|\[\.+\]'
    r'|\[\s*x\s*\]|\(\s*x\s*\)'
    r'|(?<!\w)x{2,}(?!\w)'
    r'|(?<!\w)x(?!\w)'
    r'|\(\s*large\s+break\s*\)'
    r'|\(\s*break\s*\)'
    r'|\(\s*\d+\s+broken\s+lines?\s*\)',
    re.I,
)
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')
_WS_RE        = re.compile(r'\s+')

# Determinative patterns for source preprocessing
_UNICODE_UPPER = r'A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u1E00-\u1EFF'
_UNICODE_LOWER = r'a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u1E01-\u1EFF'
_DET_UPPER_RE  = re.compile(r'\(([' + _UNICODE_UPPER + r'0-9]{1,6})\)')
_DET_LOWER_RE  = re.compile(r'\(([' + _UNICODE_LOWER + r']{1,4})\)')


def _dedup_gap(text: str) -> str:
    return _MULTI_GAP_RE.sub('<gap> ', text).strip()


def preprocess(text: str) -> str:
    """Source preprocessing: normalise, remove brackets, unify gaps."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.translate(H_MAP).translate(SUBSCRIPT_MAP)
    text = _DET_UPPER_RE.sub(r'\1', text)
    text = _DET_LOWER_RE.sub(r'{\1}', text)
    text = _GAP_UNIFIED_RE.sub(' <gap> ', text)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'[!?/]', '', text)
    text = _WS_RE.sub(' ', text).strip()
    return _dedup_gap(text)


def inject_glosses(text: str, lexicon: Dict[str, str]) -> str:
    if not text or not lexicon:
        return text
    words = text.split()
    max_glosses = min(12, max(3, math.ceil(len(words) * 0.6)))
    found, seen = [], set()
    for word in words:
        clean = re.sub(r'[.,;:!?]$', '', word.strip().lower())
        if clean in lexicon and clean not in seen:
            gloss = re.sub(r'[<>{}]', '', lexicon[clean]).strip()[:25]
            if gloss and len(gloss) > 1:
                found.append(f'{word}={gloss}')
                seen.add(clean)
    if not found or len(seen) / max(len(words), 1) < MIN_COVERAGE:
        return text
    if len(found) > max_glosses:
        found = found[:max_glosses]
    return '[' + '; '.join(found) + '] ' + text


# ============================================================
#  POSTPROCESSING
#  Key rules (confirmed from competition discussion):
#    - NO fraction conversion (0.3333 stays as 0.3333)
#    - Parentheses NOT removed (appear in test labels)
#    - Curly quotes → straight quotes (not deleted)
#    - ḫ/Ḫ → h/H in output
#    - <gap> protected via sentinel during char removal
# ============================================================

_SOFT_GRAM_RE = re.compile(
    r'\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)'
    r'(?:\.\s*(?:plur|plural|sing|singular))?\.?\s*[^)]*\)',
    re.I,
)
_BARE_GRAM_RE    = re.compile(r'(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*', re.I)
_UNCERTAIN_RE    = re.compile(r'\(\?\)')
_CURLY_DQ_RE     = re.compile('[\u201c\u201d]')   # " " → "
_CURLY_SQ_RE     = re.compile('[\u2018\u2019]')   # ' ' → '
_MONTH_RE        = re.compile(r'\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b', re.I)
_ROMAN2INT       = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,
                    'VIII':8,'IX':9,'X':10,'XI':11,'XII':12}
_STRAY_MARKS_RE  = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')

# Characters to strip — NOTE: no parens, no ḫ (handled by _HACEK_TRANS)
_FORBIDDEN_TRANS = str.maketrans('', '', '——<>⌈⌋⌊[]+ʾ;')

# Slash-alternative removal: only " / word" patterns, NOT "1/3" fractions
_SLASH_ALT_RE    = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')

_REPEAT_WORD_RE  = re.compile(r'\b(\w+)(?:\s+\1\b)+')
_REPEAT_PUNCT_RE = re.compile(r'([.,])\1+')
_PUNCT_SPACE_RE  = re.compile(r'\s+([.,:])')

# ḫ/Ḫ normalisation for model output (model trained on h/H but may still emit ḫ)
_HACEK_TRANS     = str.maketrans({'ḫ': 'h', 'Ḫ': 'H'})


def _month_repl(m: re.Match) -> str:
    return f'Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}'


def postprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''

    # 1. Gap normalisation
    text = text.replace('<big_gap>', '<gap>')
    text = _dedup_gap(text)

    # 2. NO fraction conversion — test labels keep floats (0.3333, 1.3333)
    #    Confirmed by competition host: "Policy for fractions is unchanged"

    # 3. Grammar markers (safe to remove — confirmed by host)
    text = _SOFT_GRAM_RE.sub(' ', text)
    text = _BARE_GRAM_RE.sub(' ', text)
    text = _UNCERTAIN_RE.sub('', text)

    # 4. Stray markup — but NOT parentheses (they appear in test labels)
    text = _STRAY_MARKS_RE.sub('', text)
    text = _SLASH_ALT_RE.sub('', text)

    # 5. Curly → straight quotes (NOT deletion)
    text = _CURLY_DQ_RE.sub('"', text)
    text = _CURLY_SQ_RE.sub("'", text)

    # 6. Possessive apostrophes: "Aššur's" → "Aššurs" (confirmed by host)
    text = re.sub(r"'s\b", 's', text)

    # 7. Month normalisation
    text = _MONTH_RE.sub(_month_repl, text)

    # 8. Protect <gap> during forbidden-char removal, then strip
    text = text.replace('<gap>', '\x00GAP\x00')
    text = text.translate(_FORBIDDEN_TRANS)
    text = text.replace('\x00GAP\x00', ' <gap> ')

    # 9. ḫ/Ḫ → h/H in model output
    text = text.translate(_HACEK_TRANS)

    # 10. Repetition removal — word level
    text = _REPEAT_WORD_RE.sub(r'\1', text)

    # 11. Repetition removal — phrase level (up to 6-grams)
    words = text.split()
    for plen in range(6, 1, -1):
        i, out = 0, []
        while i < len(words):
            if (i + plen * 2 <= len(words)
                    and words[i:i+plen] == words[i+plen:i+plen*2]):
                out.extend(words[i:i+plen])
                j = i + plen
                while j + plen <= len(words) and words[j:j+plen] == words[i:i+plen]:
                    j += plen
                i = j
            else:
                out.append(words[i]); i += 1
        words = out
    text = ' '.join(words)

    # 12. Final cleanup
    text = _PUNCT_SPACE_RE.sub(r'\1', text)
    text = _REPEAT_PUNCT_RE.sub(r'\1', text)
    text = _WS_RE.sub(' ', text).strip().strip('-').strip()

    return text


# ============================================================
#  MBR DECODING
# ============================================================

_chrf_metric = sacrebleu.metrics.CHRF(word_order=2)


def _chrfpp(a: str, b: str) -> float:
    a, b = (a or '').strip(), (b or '').strip()
    if not a or not b:
        return 0.0
    return float(_chrf_metric.sentence_score(a, [b]).score)


def mbr_pick(candidates: List[str]) -> str:
    seen, cands = set(), []
    for c in candidates:
        c = str(c).strip()
        if c and c not in seen:
            cands.append(c); seen.add(c)
    cands = cands[:MBR_POOL_CAP]
    n = len(cands)
    if n == 0: return ''
    if n == 1: return cands[0]
    best_i, best_s = 0, -1e9
    for i in range(n):
        s = sum(_chrfpp(cands[i], cands[j]) for j in range(n) if j != i)
        s /= max(1, n - 1)
        if s > best_s:
            best_s, best_i = s, i
    return cands[best_i]


def mbr_decode(model, tokenizer, input_text: str) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(input_text, return_tensors='pt',
                    max_length=MAX_LENGTH, truncation=True).to(device)
    tok_len = int(enc.input_ids.shape[1])
    n_beams = NUM_BEAMS_SHORT if tok_len < 60 else NUM_BEAMS

    candidates = []
    with torch.inference_mode(), _autocast_ctx():
        # Beam search — diverse candidates
        beam_outs = model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            num_beams=n_beams,
            num_return_sequences=n_beams,   # return ALL beams to MBR pool
            max_new_tokens=MAX_LENGTH,
            length_penalty=LENGTH_PENALTY,
            repetition_penalty=REPETITION_PENALTY,
            early_stopping=True,
        )
        candidates.extend(tokenizer.batch_decode(beam_outs, skip_special_tokens=True))

        # Sampling — additional diversity
        samp_outs = model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            do_sample=True,
            temperature=0.70,
            top_p=0.92,
            num_return_sequences=NUM_SAMPLES,
            max_new_tokens=MAX_LENGTH,
            repetition_penalty=REPETITION_PENALTY,
        )
        candidates.extend(tokenizer.batch_decode(samp_outs, skip_special_tokens=True))

    return mbr_pick([postprocess(c) for c in candidates if c.strip()])


# ============================================================
#  MAIN
# ============================================================

def generate_translations():
    print('=' * 60)
    print(f'Deep Past — Inference v9 Final | MBR={USE_MBR} | BF16={USE_BF16}')
    print(f'Beams: {NUM_BEAMS} (short: {NUM_BEAMS_SHORT}) | Samples: {NUM_SAMPLES}')
    print('=' * 60)

    print('\nLoading model...')
    model, tokenizer, use_glosses = load_model()

    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        print('  BetterTransformer: ON')
    except Exception:
        print('  BetterTransformer: unavailable (skipped)')

    print(f'  Parameters: {model.num_parameters():,}')

    lexicon = build_lexicon() if use_glosses else {}

    test_df = pd.read_csv(TEST_FILE)
    print(f'\nTest rows: {len(test_df)}')

    # Preprocess + gloss all inputs
    raw_texts    = test_df['transliteration'].tolist()
    clean_texts  = [preprocess(r) for r in raw_texts]
    input_texts  = [inject_glosses(c, lexicon) if use_glosses else c
                    for c in clean_texts]

    if use_glosses:
        glossed_n = sum(1 for c, g in zip(clean_texts, input_texts) if g != c)
        print(f'Gloss coverage: {glossed_n}/{len(input_texts)} '
              f'({100*glossed_n/max(len(input_texts),1):.1f}%)')

    ids     = test_df['id'].tolist()
    results = {}

    print(f'\nMBR decoding {len(ids)} rows...')
    prefixed = ['translate Akkadian to English: ' + t for t in input_texts]
    for sid, inp in tqdm(zip(ids, prefixed), total=len(ids), desc='MBR'):
        try:
            results[sid] = mbr_decode(model, tokenizer, inp)
        except Exception as e:
            print(f'  Error on {sid}: {e}')
            results[sid] = ''
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sub = (pd.DataFrame(list(results.items()), columns=['id', 'translation'])
             .sort_values('id').reset_index(drop=True))

    # Fill any empty translations
    empty = sub['translation'].str.strip().eq('').sum()
    if empty > 0:
        sub.loc[sub['translation'].str.strip() == '', 'translation'] = \
            'The tablet is too damaged to translate.'

    assert len(sub) == len(test_df), f'Row mismatch: {len(sub)} vs {len(test_df)}'
    sub.to_csv(SUBMISSION, index=False)

    print(f'\n{"="*60}')
    print(f'Saved: {SUBMISSION} | Rows: {len(sub)} | Empty filled: {empty}')
    avg = sub['translation'].str.split().str.len().mean()
    print(f'Avg words/translation: {avg:.1f}')
    print('\nSample outputs:')
    for _, row in sub.head(5).iterrows():
        print(f'  [{str(row["id"]):>6}] {str(row["translation"])[:90]}')
    print('=' * 60)


if __name__ == '__main__':
    generate_translations()

# %% [code]
