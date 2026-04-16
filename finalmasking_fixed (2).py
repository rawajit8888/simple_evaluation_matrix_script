"""
Entity-Based Masking Script — NUMBERED TAG VERSION
====================================================
Architecture
------------
  • IndicBERT NER  →  PERSON, EMAIL_ADDRESS, CORP_ID, USER_ID, ORG (via model tags)
  • Regex patterns →  PHONE, BANK_ACC, AADHAR, PAN, IFSC, CIF, DP_ACCOUNT,
                      DP_ACCOUNT_CDSL, DP_ID, TRADING_ID, IP_ADDRESS, URL

Design decisions
-----------------
  • Presidio AnalyzerEngine is used WITHOUT any spaCy/stanza NLP engine.
    A lightweight no-op NlpEngine is injected so the analyzer never tries to
    load en_core_web_lg (which most machines do not have installed).
    THIS WAS THE #1 REASON NOTHING WAS BEING MASKED BEFORE.
  • IndicNERRecognizer maps every tag the model can emit — not just PER.
  • Regex patterns use NO {min,max} quantifiers on account numbers so that
    any length of digit string is caught (fixes missed 11-digit account).
  • CORP_ID regex has NO length constraint as instructed.
  • Overlap resolution: greedy best-first (highest score then longest span).
  • Replacement: right-to-left so character offsets stay valid.

Token format:  [label N]   e.g. [name 1], [account number 2], [mobile 1]
"""

import re
import json
import os
import ssl
import requests
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import (
    Pattern, PatternRecognizer, AnalyzerEngine,
    RecognizerRegistry, EntityRecognizer, RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngine, NlpArtifacts

# ── SSL fix ───────────────────────────────────────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"]    = ""
ssl._create_default_https_context = ssl._create_unverified_context

# ── Device ────────────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")


# =============================================================================
# 1.  NO-OP NLP ENGINE
#     Presidio always calls nlp_engine.process_text() before running any
#     recognizer.  Without this shim it tries to load spaCy en_core_web_lg
#     and either crashes (model not installed) or returns zero results.
# =============================================================================
class NoOpNlpEngine(NlpEngine):
    """Minimal NlpEngine that satisfies Presidio without loading spaCy/stanza."""

    is_loaded = True

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        return NlpArtifacts(
            entities=[],
            tokens=[],
            tokens_indices=[],
            dependencies=[],
            lemmas=[],
            keywords=[],
            language=language,
        )

    def process_batch(self, texts, language: str):
        return [self.process_text(t, language) for t in texts]

    def is_stopword(self, word: str, language: str) -> bool:
        return False

    def get_supported_languages(self) -> List[str]:
        return ["en"]

    def load(self):
        pass


# =============================================================================
# 2.  ENTITY → HUMAN-READABLE LABEL
# =============================================================================
ENTITY_LABELS = {
    # ── IndicNER detected ────────────────────────────────────────────────────
    "PERSON":           "name",
    "EMAIL_ADDRESS":    "email",
    "PHONE_NUMBER":     "mobile",
    "CORP_ID":          "corp id",
    "USER_ID":          "user id",
    "ORG":              "organisation",
    # ── Regex detected ──────────────────────────────────────────────────────
    "BANK_ACC":         "account number",
    "DP_ACCOUNT":       "account number",
    "DP_ACCOUNT_CDSL":  "account number",
    "AADHAR":           "aadhaar",
    "PAN":              "pan",
    "IFSC":             "ifsc",
    "CIF":              "cif number",
    "TRADING_ID":       "trading id",
    "DP_ID":            "dp id",
    "IP_ADDRESS":       "ip address",
    "URL":              "url",
}

def make_token(label: str, n: int) -> str:
    return f"[{label} {n}]"


# =============================================================================
# 3.  HTML UTILITIES
# =============================================================================
def strip_html(html_text: str) -> str:
    """Strip HTML tags → plain text, preserving word boundaries."""
    plain = re.sub(r'<[^>]+>', ' ', html_text)
    plain = plain.replace('&nbsp;', ' ').replace('&amp;', '&')
    plain = plain.replace('&lt;', '<').replace('&gt;', '>')
    return re.sub(r'\s+', ' ', plain).strip()


def replace_in_html(html_text: str, word: str, token: str) -> str:
    """
    Replace the FIRST occurrence of `word` inside HTML text nodes only
    (never inside tag attributes or tag names).
    """
    escaped  = re.escape(word)
    pattern  = r'(<[^>]*>)|(' + escaped + r')'
    replaced = [False]

    def replacer(m):
        if m.group(1):          # inside a tag — leave unchanged
            return m.group(1)
        if not replaced[0]:     # first text match — replace
            replaced[0] = True
            return token
        return m.group(2)       # later matches — leave unchanged

    return re.sub(pattern, replacer, html_text, flags=re.IGNORECASE)


# =============================================================================
# 4.  INDICBERT NER PIPELINE  +  RECOGNIZER
# =============================================================================
MODEL_PATH = r"E:\FeatSystems\masking_data_label_studio\indicmodel"

tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model_ner     = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)

ner_pipeline = pipeline(
    "ner",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    device=device,
)

# Map every tag your IndicBERT model emits → Presidio entity type.
# If your fine-tuned model uses different tag names, add them here.
INDIC_TAG_TO_ENTITY: dict = {
    # Person names
    "PER":            "PERSON",
    "PERSON":         "PERSON",
    "B-PER":          "PERSON",
    "I-PER":          "PERSON",
    # Organisations / companies
    "ORG":            "ORG",
    "B-ORG":          "ORG",
    "I-ORG":          "ORG",
    # Email addresses
    "EMAIL":          "EMAIL_ADDRESS",
    "B-EMAIL":        "EMAIL_ADDRESS",
    "I-EMAIL":        "EMAIL_ADDRESS",
    # Phone numbers
    "PHONE":          "PHONE_NUMBER",
    "B-PHONE":        "PHONE_NUMBER",
    "I-PHONE":        "PHONE_NUMBER",
    # Corporate / company IDs
    "CID":            "CORP_ID",
    "CORPID":         "CORP_ID",
    "CORP_ID":        "CORP_ID",
    # User IDs
    "UID":            "USER_ID",
    "USERID":         "USER_ID",
    "USER_ID":        "USER_ID",
}

INDIC_SUPPORTED_ENTITIES: List[str] = list(set(INDIC_TAG_TO_ENTITY.values()))


class IndicNERRecognizer(EntityRecognizer):
    """
    Wraps the local IndicBERT NER pipeline as a Presidio recognizer.
    Covers ALL entity types the model can emit — not just PERSON.
    """

    def __init__(self, ner_pipe):
        self._pipe = ner_pipe
        super().__init__(
            supported_entities=INDIC_SUPPORTED_ENTITIES,
            name="IndicNERRecognizer",
        )

    def load(self):
        pass  # model already loaded above

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        results: List[RecognizerResult] = []
        wanted = set(entities) & set(INDIC_SUPPORTED_ENTITIES)
        if not wanted:
            return results

        try:
            predictions = self._pipe(text)
        except Exception as exc:
            print(f"  [IndicNER] pipeline error: {exc}")
            return results

        for p in predictions:
            raw_tag     = p.get("entity_group", p.get("entity", ""))
            entity_type = INDIC_TAG_TO_ENTITY.get(raw_tag)
            if entity_type is None or entity_type not in wanted:
                continue
            results.append(RecognizerResult(
                entity_type=entity_type,
                start=p["start"],
                end=p["end"],
                score=float(p["score"]),
            ))
        return results


# =============================================================================
# 5.  REGEX PATTERNS
#     Loaded from config.json when present; otherwise built-in fallbacks used.
# =============================================================================
CONFIG_PATH = r"E:\FeatSystems\masking_data_label_studio\config.json"

# Built-in fallback patterns.
# Rules applied:
#   • Account numbers: NO {min,max} range — catches any length.
#   • CORP_ID: NO length constraint — any alphanumeric value after keyword.
#   • TRADING_ID: negative lookaround — no false match inside longer numbers.
#   • AADHAR: word-boundary anchored properly.
BUILTIN_PATTERNS: List[tuple] = [
    # ── Account numbers (most-specific first for overlap resolver) ─────────
    ("DP_ACCOUNT_CDSL",  r"\b\d{16}\b"),           # exactly 16 digits
    ("DP_ACCOUNT",       r"\b\d{8}\b"),             # exactly 8 digits
    ("BANK_ACC",         r"\b\d{9,}\b"),            # 9+ digits (11, 14, 18 …)

    # ── DP ID ──────────────────────────────────────────────────────────────
    ("DP_ID",            r"\bIN\s*\d+\b"),

    # ── Trading ID — 6-7 digits, not part of a longer number ──────────────
    ("TRADING_ID",       r"(?<!\d)\d{6,7}(?!\d)"),

    # ── PAN card ───────────────────────────────────────────────────────────
    ("PAN",              r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),

    # ── Aadhaar — 12 digits, optionally space-separated (4 4 4) ───────────
    ("AADHAR",           r"\b\d{4}\s?\d{4}\s?\d{4}\b"),

    # ── IFSC ───────────────────────────────────────────────────────────────
    ("IFSC",             r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),

    # ── CIF — keyword required, digits of any length ───────────────────────
    ("CIF",
     r"\b(?:CIF|cif)\s*(?:no\.?|number|#)?\s*[:\-]?\s*\d+\b"),

    # ── Corporate ID — keyword required, alphanumeric, NO length limit ──────
    ("CORP_ID",
     r"\b(?:CORP(?:\s*ID)?|corp(?:\s*id)?|CORPID|corpid)\s*[:\-#]?\s*[A-Z0-9]+\b"),

    # ── User ID — keyword required, alphanumeric, any length ────────────────
    ("USER_ID",
     r"\b(?:USER\s*ID|user\s*id|UID|uid)\s*[:\-#]?\s*[A-Z0-9]+\b"),

    # ── IP Address ─────────────────────────────────────────────────────────
    ("IP_ADDRESS",
     r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),

    # ── Email (regex fallback — IndicNER is primary source) ─────────────────
    ("EMAIL_ADDRESS",
     r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),

    # ── Indian mobile — 10 digits starting with 6-9 ────────────────────────
    ("PHONE_NUMBER",
     r"(?<!\d)[6-9]\d{9}(?!\d)"),
]

patterns: List[tuple] = []
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        config_data = json.load(fh)
    patterns = [
        (item["name"], item["regex"])
        for item in config_data.get("patterns", [])
    ]
    print(f"Loaded {len(patterns)} regex patterns from {CONFIG_PATH}")
else:
    print(f"WARNING: config.json not found at {CONFIG_PATH} — using built-in patterns")
    patterns = BUILTIN_PATTERNS


# =============================================================================
# 6.  BUILD PRESIDIO REGISTRY  (no spaCy, no predefined recognizers)
# =============================================================================
registry = RecognizerRegistry()
# Do NOT call registry.load_predefined_recognizers()
# That method loads SpacyRecognizer which:
#   (a) conflicts with IndicNERRecognizer for PERSON
#   (b) crashes if en_core_web_lg is not installed

for ent_name, regex in patterns:
    pat = Pattern(name=f"{ent_name}_pat", regex=regex, score=0.85)
    registry.add_recognizer(
        PatternRecognizer(supported_entity=ent_name, patterns=[pat])
    )

registry.add_recognizer(IndicNERRecognizer(ner_pipeline))

analyzer = AnalyzerEngine(
    registry=registry,
    nlp_engine=NoOpNlpEngine(),
    supported_languages=["en"],
)

ALL_ENTITIES: List[str] = list(ENTITY_LABELS.keys())


# =============================================================================
# 7.  OVERLAP RESOLVER
# =============================================================================
def resolve_overlaps(results: list) -> list:
    """
    Greedy best-first: keep highest-score / longest span; discard anything
    that overlaps an already-kept span.  Return sorted by start position.
    """
    best_first = sorted(results, key=lambda r: (-r.score, -(r.end - r.start)))
    kept: list = []
    for r in best_first:
        if not any(not (r.end <= k.start or r.start >= k.end) for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r.start)


# =============================================================================
# 8.  CORE MASKING FUNCTION
# =============================================================================
def mask_text(text: str) -> str:
    """
    Detect all PII in `text` and replace each with a numbered readable token.

    Plain-text examples
    -------------------
    IN : "Unblock account 12345678901, phone 9892333333"
    OUT: "Unblock account [account number 1], phone [mobile 1]"

    IN : "Dear Rajesh Kumar, Corp ID: ABCXYZ123"
    OUT: "Dear [name 1], Corp ID: [corp id 1]"

    HTML example
    ------------
    IN : "<p><b>Priya Sharma</b> account <span>12345678901</span></p>"
    OUT: "<p><b>[name 1]</b> account <span>[account number 1]</span></p>"
    """
    if not text or not text.strip():
        return text

    # Step 1 — strip HTML for NER analysis
    is_html = bool(re.search(r'<[a-zA-Z]', text))
    plain   = strip_html(text) if is_html else text
    if not plain.strip():
        return text

    # Step 2 — run all recognizers on plain text
    try:
        raw_results = analyzer.analyze(
            text=plain,
            entities=ALL_ENTITIES,
            language="en",
        )
    except Exception as exc:
        print(f"  [mask_text] analyzer error: {exc}")
        return text

    if not raw_results:
        return text

    # Step 3 — resolve overlapping detections
    filtered = resolve_overlaps(raw_results)

    # Step 4 — assign numbered tokens left-to-right
    label_counters: dict = {}
    assignments:    list = []   # (result, original_word, token)

    for r in filtered:          # already sorted by start
        label = ENTITY_LABELS.get(
            r.entity_type,
            r.entity_type.lower().replace("_", " "),
        )
        label_counters[label] = label_counters.get(label, 0) + 1
        token         = make_token(label, label_counters[label])
        original_word = plain[r.start:r.end]
        assignments.append((r, original_word, token))

    # Step 5a — plain text: replace right-to-left (keeps offsets valid)
    if not is_html:
        masked = plain
        for r, _, token in reversed(assignments):
            masked = masked[:r.start] + token + masked[r.end:]
        return masked

    # Step 5b — HTML: replace each word in HTML text nodes only
    result_html = text
    for _, original_word, token in reversed(assignments):
        result_html = replace_in_html(result_html, original_word, token)
    return result_html


# =============================================================================
# 9.  QUICK SELF-TEST
# =============================================================================
def run_test():
    test_cases = [
        # Account numbers — various lengths
        ("plain", "My account number is 12345678901 please unblock"),
        ("plain", "Account 98765432 blocked"),
        ("plain", "Account 1234567890123456 under review"),
        # Phone
        ("plain", "Call me on 9892333333 or 9811122233"),
        # Email
        ("plain", "Send to user@example.com please"),
        # PAN + Aadhaar
        ("plain", "PAN ABCDE1234F, Aadhaar 1234 5678 9012"),
        # IFSC + CIF
        ("plain", "IFSC HDFC0001234, CIF: 00123456"),
        # Corp ID — no length constraint
        ("plain", "Corp ID: ABC1234XYZ and CORPID XY99"),
        # User ID
        ("plain", "User ID: UID456789"),
        # DP ID
        ("plain", "DP ID IN123456789"),
        # Name via IndicNER
        ("plain", "Dear Rajesh Kumar, your account is blocked"),
        # Multiple same-label entities
        ("plain", "Call 9892333333 or 9811122233"),
        # HTML
        ("html",
         "<p>Dear <b>Priya Sharma</b>, account <span>12345678901</span> "
         "CIF: <b>00123456</b> CORP ID: <b>ABCXYZ123</b></p>"),
    ]

    print("\n" + "=" * 72)
    print("MASKING SELF-TEST")
    print("=" * 72)
    for kind, text in test_cases:
        masked = mask_text(text)
        status = "MASKED   " if masked != text else "UNCHANGED"
        print(f"\n  [{status}] {kind.upper()}")
        print(f"  IN : {text}")
        print(f"  OUT: {masked}")
    print("=" * 72 + "\n")


# =============================================================================
# 10.  LABEL STUDIO INTEGRATION
# =============================================================================
API_TOKEN = ""                       # ← fill this in
BASE_URL  = "http://localhost:8080/api"
PROXIES   = {"http": None}


def get_tasks(project_id: int, page: int = 1) -> list:
    url     = f"{BASE_URL}/tasks/?page={page}&project={project_id}&fields=task_only"
    headers = {"accept": "application/json", "Authorization": f"Token {API_TOKEN}"}
    try:
        r = requests.get(url, headers=headers, proxies=PROXIES, verify=False)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data.get("results", []) or data.get("tasks", [])
        return data
    except requests.exceptions.RequestException as exc:
        print(f"  [get_tasks] page {page} error: {exc}")
        return []


def update_task(task: dict) -> None:
    task_id       = task["id"]
    original_html = task.get("data", {}).get("html", "")

    if not original_html:
        print(f"  Task {task_id}: no html field — skipping")
        return

    masked_html = mask_text(original_html)

    if masked_html == original_html:
        print(f"  Task {task_id}: no entities found — skipped")
        return

    task["data"]["html"] = masked_html
    url     = f"{BASE_URL}/tasks/{task_id}/"
    headers = {
        "accept":        "application/json",
        "Content-Type":  "application/json",
        "Authorization": f"Token {API_TOKEN}",
    }
    resp = requests.patch(
        url,
        headers=headers,
        proxies=PROXIES,
        json={"data": task["data"], "project": task["project"]},
        verify=False,
    )
    if resp.status_code in (200, 201):
        print(f"  Task {task_id}: masked and updated ✓")
    else:
        print(f"  Task {task_id}: FAILED {resp.status_code} — {resp.text[:120]}")


def process_project(project_id: int) -> None:
    page, total = 1, 0
    print(f"Processing project {project_id}...")
    while True:
        tasks = get_tasks(project_id, page=page)
        if not tasks:
            break
        print(f"\nPage {page} — {len(tasks)} tasks")
        for task in tasks:
            update_task(task)
            total += 1
        page += 1
    print(f"\nDone. Total tasks processed: {total}")


# =============================================================================
# 11.  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_test()
    PROJECT_ID = 16
    process_project(PROJECT_ID)
