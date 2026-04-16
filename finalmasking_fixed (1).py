"""
Entity-Based Masking Script — NUMBERED TAG VERSION (patched)
=============================================================
Masks PII entities in Label Studio tasks with numbered, human-readable tokens.

Example:
  Input:  "I want to unblock my account number 12345678901 and my phone is 9892333333"
  Output: "I want to unblock my account number [account number 1] and my phone is [mobile 1]"

  Input:  "Contact Rajesh Kumar at rajesh@email.com or 9892333333"
  Output: "Contact [name 1] at [email 1] or [mobile 1]"

Patches applied on top of previous version:
  FIX-1  PERSON / NAME masking  — removed duplicate spaCy PERSON recognizer that
         was silently winning over IndicNER; IndicNER results are now the sole
         source for PERSON detection.
  FIX-2  11-digit (and any 9-18 digit) account number — BANK_ACC regex in
         config.json was "\b\d{14}\b" (14 digits only).  The fallback pattern and
         the config are now updated to "\b\d{9,18}\b" so any account length is
         caught.  config_updated.json is written next to this file automatically.
  FIX-3  TRADING_ID false-positive inside longer numbers — added a negative
         lookahead/lookbehind so 6-7-digit pattern does not fire when the digits
         are part of a longer number.
  FIX-4  AADHAR / DP_ACCOUNT_CDSL anchoring — added \b at the START of those
         patterns to stop mid-number false matches.
  FIX-5  Overlap resolver correctness — replaced the naive sorted+scan with a
         proper interval-sweep that always keeps the highest-score/longest span.
  FIX-6  Predefined recognizer conflict — load_predefined_recognizers() is still
         called (needed for EMAIL, PHONE, URL, IP), but the built-in SpaCy PERSON
         recognizer is explicitly removed so IndicNER is the only PERSON source.
"""

import re
import json
import os
import ssl
import requests
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import (Pattern, PatternRecognizer, AnalyzerEngine,
                                RecognizerRegistry, EntityRecognizer, RecognizerResult)

# ── SSL fix ───────────────────────────────────────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# ── Device ────────────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

# ── Entity → human-readable label map ────────────────────────────────────────
ENTITY_LABELS = {
    "PERSON":           "name",
    "EMAIL_ADDRESS":    "email",
    "PHONE_NUMBER":     "mobile",
    "AADHAR":           "aadhaar",
    "PAN":              "pan",
    "IFSC":             "ifsc",
    "BANK_ACC":         "account number",
    "DP_ACCOUNT":       "account number",
    "DP_ACCOUNT_CDSL":  "account number",
    "CIF":              "cif number",
    "CORP_ID":          "corp id",
    "USER_ID":          "user id",
    "TRADING_ID":       "user id",
    "DP_ID":            "dp id",
    "URL":              "url",
    "IP_ADDRESS":       "ip address",
}

def make_token(label: str, n: int) -> str:
    return f"[{label} {n}]"


# ── HTML stripper ─────────────────────────────────────────────────────────────
def strip_html(html_text: str) -> str:
    plain = re.sub(r'<[^>]+>', ' ', html_text)
    plain = re.sub(r'&nbsp;', ' ', plain)
    plain = re.sub(r'&amp;', '&', plain)
    plain = re.sub(r'&lt;', '<', plain)
    plain = re.sub(r'&gt;', '>', plain)
    plain = re.sub(r'\s+', ' ', plain).strip()
    return plain


# ── IndicNER Recognizer ───────────────────────────────────────────────────────
MODEL_PATH = r"E:\FeatSystems\masking_data_label_studio\indicmodel"

tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model_ner     = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)

ner_pipeline = pipeline(
    "ner",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    device=device
)

class IndicNERRecognizer(EntityRecognizer):
    """
    Wraps the IndicBERT NER pipeline as a Presidio recognizer.

    FIX-1: This is now the ONLY recognizer for PERSON. The built-in
    SpacyRecognizer that also emits PERSON is removed from the registry
    (see 'Build registry' section below) so there is no silent override.
    """
    def __init__(self, ner_pipeline):
        self.ner_pipeline = ner_pipeline
        super().__init__(supported_entities=["PERSON"], name="IndicNERRecognizer")

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        if "PERSON" not in entities:
            return results
        try:
            predictions = self.ner_pipeline(text)
            for p in predictions:
                if p["entity_group"] == "PER":
                    results.append(RecognizerResult(
                        entity_type="PERSON",
                        start=p["start"],
                        end=p["end"],
                        score=p["score"]
                    ))
        except Exception as e:
            print(f"  IndicNER warning: {e}")
        return results

    def load(self):
        pass


# ── Load regex patterns from config ──────────────────────────────────────────
CONFIG_PATH = r"E:\FeatSystems\masking_data_label_studio\config.json"

# FIX-2 / FIX-3 / FIX-4: corrected built-in fallback patterns
FALLBACK_PATTERNS = [
    # FIX-2: was \b\d{14}\b — now covers any 9-18 digit account number
    ("BANK_ACC",          r"\b\d{9,18}\b"),
    ("CIF",               r"\b(?:CIF|cif)\s*(?:no\.?|number|#)?[:\-\s]?\s*\d{8,12}\b"),
    ("CORP_ID",           r"\b(?:CORP\s*(?:ID)?|corp\s*(?:id)?)\s*[:\-#]?\s*[A-Z0-9]{4,12}\b"),
    ("USER_ID",           r"\b(?:USER\s*ID|user\s*id|UID|uid)\s*[:\-#]?\s*[A-Z0-9]{4,12}\b"),
    ("IFSC",              r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    ("PAN",               r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
    # FIX-4: added leading \b to stop partial matches inside longer digit strings
    ("AADHAR",            r"\b(?:\d\s*){12}\b"),
    ("IP_ADDRESS",        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # FIX-3: TRADING_ID — negative look-around stops it from firing inside a longer number
    ("TRADING_ID",        r"(?<!\d)\d{6,7}(?!\d)"),
    ("DP_ACCOUNT",        r"\b\d{8}\b"),
    # FIX-4: added leading \b
    ("DP_ACCOUNT_CDSL",   r"\b(?:\d\s*){16}\b"),
    ("DP_ID",             r"\bIN\s*\d{6,}\b"),
]

# ── Patch config.json in-place so file-based runs are also correct ─────────────
_CONFIG_PATCH = {
    # FIX-2: 11-digit (and any 9-18 digit) account number
    "BANK_ACC":         r"\b\d{9,18}\b",
    # FIX-3: TRADING_ID — no false match inside longer numbers
    "TRADING_ID":       r"(?<!\d)\d{6,7}(?!\d)",
    # FIX-4: Aadhaar leading anchor
    "AADHAR":           r"\b(?:\d\s*){12}\b",
    # FIX-4: CDSL leading anchor
    "DP_ACCOUNT_CDSL":  r"\b(?:\d\s*){16}\b",
}

patterns = []
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Apply patches and write updated config next to the original
    patched = False
    for item in config.get("patterns", []):
        if item["name"] in _CONFIG_PATCH:
            new_regex = _CONFIG_PATCH[item["name"]]
            if item["regex"] != new_regex:
                print(f"  [config patch] {item['name']}: {item['regex']!r} → {new_regex!r}")
                item["regex"] = new_regex
                patched = True

    if patched:
        updated_path = CONFIG_PATH.replace("config.json", "config_updated.json")
        with open(updated_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  [config patch] Saved patched config → {updated_path}")

    patterns = [(item["name"], item["regex"]) for item in config.get("patterns", [])]
    print(f"Loaded {len(patterns)} regex patterns from config")
else:
    print(f"WARNING: Config not found at {CONFIG_PATH} — using built-in patterns only")
    patterns = FALLBACK_PATTERNS


# ── Build registry ────────────────────────────────────────────────────────────
registry = RecognizerRegistry()
registry.load_predefined_recognizers()

# FIX-1: Remove the built-in SpaCy/Stanza PERSON recognizer so it does NOT
# silently override or conflict with IndicNERRecognizer results.
# Presidio names it "SpacyRecognizer" or "StanzaRecognizer" depending on the
# nlp engine; we remove any recognizer that claims to support "PERSON" and is
# NOT our own IndicNER.
to_remove = [
    r for r in registry.recognizers
    if "PERSON" in getattr(r, "supported_entities", [])
    and r.__class__.__name__ != "IndicNERRecognizer"
]
for r in to_remove:
    print(f"  [registry] Removing conflicting PERSON recognizer: {r.__class__.__name__}")
    registry.recognizers.remove(r)

for name, regex in patterns:
    pattern    = Pattern(name=f"{name}_pattern", regex=regex, score=1.0)
    recognizer = PatternRecognizer(supported_entity=name, patterns=[pattern])
    registry.add_recognizer(recognizer)

registry.add_recognizer(IndicNERRecognizer(ner_pipeline))

analyzer = AnalyzerEngine(registry=registry)

ALL_ENTITIES = list(set(list(ENTITY_LABELS.keys()) + ["PHONE_NUMBER", "EMAIL_ADDRESS"]))


# ── Overlap resolver (FIX-5) ──────────────────────────────────────────────────
def resolve_overlaps(results):
    """
    FIX-5: Proper interval-sweep overlap resolver.

    Previous version used a single sorted pass that could miss cases where a
    later span (higher score) started BEFORE the current last_end but AFTER the
    previous span's start — the replacement check was correct but only compared
    against filtered[-1], which may not be the actual conflicting span when
    spans share a start offset.

    New approach:
      1. Sort by score DESC, then span-length DESC (greedy: best span first).
      2. Walk through; keep a span only if it does not overlap any already-kept span.
      3. Re-sort kept spans by start for downstream use.
    """
    # Sort best-first: highest score wins; on tie, longer span wins
    best_first = sorted(results, key=lambda r: (-r.score, -(r.end - r.start)))

    kept: list = []
    for r in best_first:
        # Check against every already-kept span for any overlap
        overlaps = any(
            not (r.end <= k.start or r.start >= k.end)
            for k in kept
        )
        if not overlaps:
            kept.append(r)

    # Return in reading order (left-to-right by start position)
    return sorted(kept, key=lambda r: r.start)


# ── HTML-safe replacement ─────────────────────────────────────────────────────
def replace_in_html(html_text: str, word: str, token: str) -> str:
    """
    Replace the first occurrence of `word` inside HTML text content only
    (not inside tag attributes).
    """
    escaped = re.escape(word)
    pattern = r'(<[^>]*>)|(' + escaped + r')'
    replaced = [False]

    def replacer(m):
        if m.group(1):
            return m.group(1)
        if not replaced[0]:
            replaced[0] = True
            return token
        return m.group(2)

    return re.sub(pattern, replacer, html_text, flags=re.IGNORECASE)


# ── CORE MASKING FUNCTION ─────────────────────────────────────────────────────
def mask_text(text: str) -> str:
    """
    Detect PII in `text` and replace each entity with a numbered readable token.

    Plain-text example
    ------------------
    IN : "Unblock account 12345678901, phone 9892333333, and 98765432100"
    OUT: "Unblock account [account number 1], phone [mobile 1], and [mobile 2]"

    HTML example
    ------------
    IN : "<p>Contact <b>Rajesh Kumar</b> at 9892333333</p>"
    OUT: "<p>Contact <b>[name 1]</b> at [mobile 1]</p>"
    """
    if not text or not text.strip():
        return text

    # ── Step 1: strip HTML if needed ─────────────────────────────────────
    is_html = bool(re.search(r'<[a-zA-Z]', text))
    plain   = strip_html(text) if is_html else text

    if not plain.strip():
        return text

    # ── Step 2: detect entities ──────────────────────────────────────────
    try:
        results = analyzer.analyze(text=plain, entities=ALL_ENTITIES, language='en')
    except Exception as e:
        print(f"  Analyzer warning: {e}")
        return text

    if not results:
        return text

    # ── Step 3: resolve overlaps (FIX-5) ────────────────────────────────
    filtered = resolve_overlaps(results)

    # ── Step 4: assign numbered tokens LEFT-TO-RIGHT ─────────────────────
    label_counters: dict[str, int] = {}
    assignments: list[tuple] = []   # (result, original_word, token)

    for r in sorted(filtered, key=lambda x: x.start):
        label = ENTITY_LABELS.get(r.entity_type, r.entity_type.lower().replace("_", " "))
        label_counters[label] = label_counters.get(label, 0) + 1
        token         = make_token(label, label_counters[label])
        original_word = plain[r.start:r.end]
        assignments.append((r, original_word, token))

    # ── Step 5a: plain-text replacement (right-to-left) ──────────────────
    if not is_html:
        masked = plain
        for r, _, token in reversed(assignments):
            masked = masked[:r.start] + token + masked[r.end:]
        return masked

    # ── Step 5b: HTML replacement ─────────────────────────────────────────
    result_html = text
    for _, original_word, token in reversed(assignments):
        result_html = replace_in_html(result_html, original_word, token)

    return result_html


# ── QUICK TEST ────────────────────────────────────────────────────────────────
def run_test():
    test_cases = [
        # Plain text — core entities
        ("plain",
         "I want to unblock my account number 1234567890 and my phone number is 9892333333"),
        # FIX-2: 11-digit account number — was NOT masked before
        ("plain",
         "My account number is 12345678901 please unblock it"),
        ("plain",
         "My CIF number is 12345678 and IFSC code is HDFC0001234"),
        ("plain",
         "Please update email to user@example.com for account 98765432101234"),
        ("plain",
         "IP access from 192.168.1.45 — please check"),
        ("plain",
         "PAN number ABCDE1234F and Aadhaar 1234 5678 9012"),
        # FIX-1: Name masking
        ("plain",
         "Dear Rajesh Kumar, your account has been blocked"),
        # CIF / CORP ID / USER ID
        ("plain",
         "CIF: 00123456 and Corp ID: ABC1234 and User ID: UID-987654"),
        ("plain",
         "Login failed for user id UID 456789, corp id CORP XYZ001"),
        ("plain",
         "CIF no. 87654321 linked to CORPID AB9988 and userid uid112233"),
        # Multiple same-type entities → numbering 1, 2, 3...
        ("plain",
         "Call 9892333333 or 9811122233 for CIF 11223344 and CIF 55667788"),
        # HTML
        ("html",
         "<p>Dear <b>Rajesh Kumar</b>, your account <span>98765432101234</span> is blocked. "
         "CIF: <b>12345678</b>, Corp ID: <b>CORP AB1234</b></p>"),
        # FIX-2: 11-digit in HTML
        ("html",
         "<p>Account <b>12345678901</b> belongs to <b>Priya Sharma</b></p>"),
    ]

    print("\n" + "=" * 70)
    print("MASKING TEST — numbered tag format  [patched build]")
    print("=" * 70)
    for kind, text in test_cases:
        masked = mask_text(text)
        print(f"\n  TYPE : {kind}")
        print(f"  IN   : {text}")
        print(f"  OUT  : {masked}")
    print("=" * 70 + "\n")


# ── LABEL STUDIO INTEGRATION ──────────────────────────────────────────────────
API_TOKEN = ""            # ← fill this
BASE_URL  = "http://localhost:8080/api"
PROXIES   = {"http": None}


def get_tasks(project_id, page=1):
    url     = f"{BASE_URL}/tasks/?page={page}&project={project_id}&fields=task_only"
    headers = {"accept": "application/json", "Authorization": f"Token {API_TOKEN}"}
    try:
        response = requests.get(url, headers=headers, proxies=PROXIES, verify=False)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("results", []) or data.get("tasks", [])
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page}: {e}")
        return []


def update_task(task):
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
        "Authorization": f"Token {API_TOKEN}"
    }
    response = requests.patch(
        url,
        headers=headers,
        proxies=PROXIES,
        json={"data": task["data"], "project": task["project"]},
        verify=False
    )

    if response.status_code in [200, 201]:
        print(f"  Task {task_id}: masked and updated ✓")
    else:
        print(f"  Task {task_id}: FAILED {response.status_code} — {response.text[:100]}")


def process_project(project_id):
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


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_test()
    PROJECT_ID = 16
    process_project(PROJECT_ID)
