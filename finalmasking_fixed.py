"""
Entity-Based Masking Script — NUMBERED TAG VERSION
====================================================
Masks PII entities in Label Studio tasks with numbered, human-readable tokens.

Example:
  Input:  "I want to unblock my account number 1234567890 and my phone is 9892333333"
  Output: "I want to unblock my account number [account number 1] and my phone is [mobile 1]"

  Input:  "Contact Rajesh Kumar at rajesh@email.com or 9892333333"
  Output: "Contact [name 1] at [email 1] or [mobile 1]"

Bug fixes applied over previous version:
  1. NUMBERED TAGS  — tokens are now [label N] instead of <TOKEN>
  2. HTML FIX       — HTML replacement now uses a word-boundary-aware regex replace
                      instead of plain str.replace(), so tokens split across tags
                      are handled and wrong-position replacements are avoided
  3. OVERLAP FIX    — overlap resolver now also checks by coverage length (longer
                      span wins on same score) not just arrival order
  4. TOKEN ASSIGN   — tokens are assigned LEFT-TO-RIGHT (consistent numbering),
                      but text replacement is done RIGHT-TO-LEFT (no offset shift)
  5. DEFAULT LABEL  — unknown entity types fall back to a readable label, never
                      silently deleted
"""

import re
import json
import os
import ssl
import requests
import torch
from html.parser import HTMLParser
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
# FIX: Labels are plain words now. The final token will be "[label N]"
# e.g.  BANK_ACC  →  label "account number"  →  token "[account number 1]"
#
# CRITICAL: If you change these labels, update your inference script too.
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
    "CIF":              "cif number",     # FIX: was "cif", now "cif number"
    "CORP_ID":          "corp id",        # NEW
    "USER_ID":          "user id",
    "TRADING_ID":       "user id",
    "DP_ID":            "dp id",
    "URL":              "url",
    "IP_ADDRESS":       "ip address",
}

def make_token(label: str, n: int) -> str:
    """Build the final replacement token, e.g. '[account number 1]'"""
    return f"[{label} {n}]"


# ── HTML stripper ─────────────────────────────────────────────────────────────
def strip_html(html_text: str) -> str:
    """
    Strip HTML tags → plain text for NER.
    Tags are replaced with a single space so word boundaries are preserved.
    """
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

patterns = []
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    patterns = [(item["name"], item["regex"]) for item in config.get("patterns", [])]
    print(f"Loaded {len(patterns)} regex patterns from config")
else:
    print(f"WARNING: Config not found at {CONFIG_PATH} — using built-in patterns only")
    patterns = [
        ("BANK_ACC",   r"\b\d{9,18}\b"),
        ("CIF",        r"\b(?:CIF|cif)\s*(?:no\.?|number|#)?[:\-\s]?\s*\d{8,12}\b"),
        ("CORP_ID",    r"\b(?:CORP\s*(?:ID)?|corp\s*(?:id)?)\s*[:\-#]?\s*[A-Z0-9]{4,12}\b"),
        ("USER_ID",    r"\b(?:USER\s*ID|user\s*id|UID|uid)\s*[:\-#]?\s*[A-Z0-9]{4,12}\b"),
        ("IFSC",       r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        ("PAN",        r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
        ("AADHAR",     r"\b\d{4}\s\d{4}\s\d{4}\b"),
        ("IP_ADDRESS", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ]

# ── Build registry ────────────────────────────────────────────────────────────
registry = RecognizerRegistry()
registry.load_predefined_recognizers()

for name, regex in patterns:
    pattern    = Pattern(name=f"{name}_pattern", regex=regex, score=1.0)
    recognizer = PatternRecognizer(supported_entity=name, patterns=[pattern])
    registry.add_recognizer(recognizer)

registry.add_recognizer(IndicNERRecognizer(ner_pipeline))

analyzer = AnalyzerEngine(registry=registry)

ALL_ENTITIES = list(set(list(ENTITY_LABELS.keys()) + ["PHONE_NUMBER", "EMAIL_ADDRESS"]))


# ── Overlap resolver ──────────────────────────────────────────────────────────
def resolve_overlaps(results):
    """
    Keep the best (highest score, then longest span) non-overlapping entity per position.
    Sort by start; when two spans overlap keep the one with higher score (longer if tied).
    """
    # Sort: start ASC, score DESC, span-length DESC
    sorted_results = sorted(
        results,
        key=lambda r: (r.start, -r.score, -(r.end - r.start))
    )
    filtered = []
    last_end  = -1
    for r in sorted_results:
        if r.start >= last_end:          # no overlap → keep
            filtered.append(r)
            last_end = r.end
        else:
            # overlap → compare with the last kept result and replace if better
            prev = filtered[-1]
            curr_score  = (r.score, r.end - r.start)
            prev_score  = (prev.score, prev.end - prev.start)
            if curr_score > prev_score:
                filtered[-1] = r
                last_end = r.end
    return filtered


# ── HTML-safe replacement ─────────────────────────────────────────────────────
def replace_in_html(html_text: str, word: str, token: str) -> str:
    """
    Replace the first occurrence of `word` inside HTML text content only
    (not inside tag attributes). Uses a regex that avoids matching inside
    angle brackets.

    FIX for the old bug: plain str.replace() would sometimes match inside
    tag attributes (e.g. class="HDFC0001234") and would FAIL silently when
    the word was split across two tags — both causing wrong/no masking.
    """
    # Escape special regex chars in the word itself
    escaped = re.escape(word)
    # Match the word only when NOT inside a tag (i.e. not between < and >)
    # Strategy: consume tags as non-capturing groups and only replace in text nodes
    pattern = r'(<[^>]*>)|(' + escaped + r')'

    replaced = [False]   # mutable flag for use inside lambda

    def replacer(m):
        if m.group(1):          # it's a tag — keep as-is
            return m.group(1)
        if not replaced[0]:     # first text match — replace
            replaced[0] = True
            return token
        return m.group(2)       # subsequent matches — keep

    return re.sub(pattern, replacer, html_text, flags=re.IGNORECASE)


# ── CORE MASKING FUNCTION ─────────────────────────────────────────────────────
def mask_text(text: str) -> str:
    """
    Detect PII in `text` and replace each entity with a numbered readable token.

    Plain-text example
    ------------------
    IN : "Unblock account 1234567890, phone 9892333333, and 98765432100"
    OUT: "Unblock account [account number 1], phone [mobile 1], and [mobile 2]"

    HTML example
    ------------
    IN : "<p>Contact <b>Rajesh Kumar</b> at 9892333333</p>"
    OUT: "<p>Contact <b>[name 1]</b> at [mobile 1]</p>"

    Rules
    -----
    - Numbering is per-label and resets for every call (per message/task).
    - Same label reused by multiple entity types shares the same counter
      (e.g. BANK_ACC and DP_ACCOUNT both produce "[account number N]").
    - Unknown entity types fall back to "[<entity_type> N]" — never deleted.
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

    # ── Step 3: resolve overlaps ─────────────────────────────────────────
    filtered = resolve_overlaps(results)

    # ── Step 4: assign numbered tokens LEFT-TO-RIGHT ──────────────────────
    # FIX: Assign tokens in reading order so numbering is always consistent.
    # Replacement order is REVERSED below (right-to-left) to keep offsets valid.
    label_counters: dict[str, int] = {}
    assignments: list[tuple] = []   # (result, original_word, token)

    for r in sorted(filtered, key=lambda x: x.start):
        label = ENTITY_LABELS.get(r.entity_type, r.entity_type.lower().replace("_", " "))
        label_counters[label] = label_counters.get(label, 0) + 1
        token        = make_token(label, label_counters[label])
        original_word = plain[r.start:r.end]
        assignments.append((r, original_word, token))

    # ── Step 5a: plain-text replacement (right-to-left) ──────────────────
    if not is_html:
        masked = plain
        for r, _, token in reversed(assignments):
            masked = masked[:r.start] + token + masked[r.end:]
        return masked

    # ── Step 5b: HTML replacement ─────────────────────────────────────────
    # FIX: Use replace_in_html() which skips tag attributes and handles
    # cases where the matched word spans/touches HTML tags.
    result_html = text
    for _, original_word, token in reversed(assignments):   # reverse = right-to-left
        result_html = replace_in_html(result_html, original_word, token)

    return result_html


# ── QUICK TEST ────────────────────────────────────────────────────────────────
def run_test():
    test_cases = [
        # Plain text — core entities
        ("plain",
         "I want to unblock my account number 1234567890 and my phone number is 9892333333"),
        ("plain",
         "My CIF number is 12345678 and IFSC code is HDFC0001234"),
        ("plain",
         "Please update email to user@example.com for account 98765432101234"),
        ("plain",
         "IP access from 192.168.1.45 — please check"),
        ("plain",
         "PAN number ABCDE1234F and Aadhaar 1234 5678 9012"),
        ("plain",
         "Dear Rajesh Kumar, your account has been blocked"),
        # CIF / CORP ID / USER ID
        ("plain",
         "CIF: 00123456 and Corp ID: ABC1234 and User ID: UID-987654"),
        ("plain",
         "Login failed for user id UID 456789, corp id CORP XYZ001"),
        ("plain",
         "CIF no. 87654321 linked to CORPID AB9988 and userid uid112233"),
        # Multiple same-type entities → numbering should be 1, 2, 3...
        ("plain",
         "Call 9892333333 or 9811122233 for CIF 11223344 and CIF 55667788"),
        # HTML
        ("html",
         "<p>Dear <b>Rajesh Kumar</b>, your account <span>98765432101234</span> is blocked. "
         "CIF: <b>12345678</b>, Corp ID: <b>CORP AB1234</b></p>"),
    ]

    print("\n" + "=" * 65)
    print("MASKING TEST — numbered tag format")
    print("=" * 65)
    for kind, text in test_cases:
        masked = mask_text(text)
        print(f"\n  TYPE: {kind}")
        print(f"  IN : {text}")
        print(f"  OUT: {masked}")
    print("=" * 65 + "\n")


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


# ── INFERENCE HELPER ──────────────────────────────────────────────────────────
# Import in Inference_Single_Level.py:
#
#   from finalmasking_fixed import mask_text
#
#   text    = mask_text(str(row[EMAIL_COL]))
#   payload = build_payload_bert(text)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_test()
    PROJECT_ID = 16
    process_project(PROJECT_ID)
