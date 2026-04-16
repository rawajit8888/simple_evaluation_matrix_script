"""
Entity-Based Masking Script — Regex + IndicNER
===============================================
All entities detected via hand-crafted regex patterns.
Only PERSON (name) detection uses the IndicNER model.
Email is also handled by regex (no Presidio needed).

Advantages over Presidio approach:
  - No overlap fights between recognizers
  - No Presidio phone recognizer eating account numbers
  - Fully transparent — what you write is what gets matched
  - Faster (no Presidio pipeline overhead)

Toggle INDEXED_TOKENS:
  False  →  <ACCOUNT>, <ACCOUNT>       (intent / classification tasks)
  True   →  <ACCOUNT_1>, <ACCOUNT_2>   (slot-fill / seq2seq tasks)
"""

import re
import json
import os
import ssl
import requests
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ── Toggle ────────────────────────────────────────────────────────────────────
INDEXED_TOKENS = False

# ── SSL fix ───────────────────────────────────────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"]    = ""
ssl._create_default_https_context = ssl._create_unverified_context

# ── Device ────────────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

# ── IndicNER — used ONLY for PERSON detection ─────────────────────────────────
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

# ── Regex patterns ────────────────────────────────────────────────────────────
# Order matters — more specific patterns must come BEFORE broader ones.
# e.g. Aadhaar (4-4-4 groups) must come before generic 9-18 digit BANK_ACC.
#
# Each entry: (entity_type, compiled_regex)
# Lookahead/lookbehind (?<!\d) / (?!\d) prevent partial digit matches.

REGEX_PATTERNS = [
    # ── Structured / high-specificity first ──────────────────────────────

    # Aadhaar: exactly 4-space-4-space-4 digit format
    ("AADHAR",       re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")),

    # PAN: AAAAA9999A format
    ("PAN",          re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),

    # IFSC: AAAA0AAAAAA format
    ("IFSC",         re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),

    # CIF: keyword + 8-12 digits
    ("CIF",          re.compile(r"\bCIF\s*[:\-]?\s*\d{8,12}\b", re.IGNORECASE)),

    # DP Account CDSL: 16 digits, optionally split 8-8
    ("DP_ACCOUNT",   re.compile(r"(?<!\d)\d{8}[-\s]?\d{8}(?!\d)")),

    # IP address: must come before generic digit patterns
    ("IP_ADDRESS",   re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),

    # Email
    ("EMAIL",        re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")),

    # URL
    ("URL",          re.compile(r"https?://[^\s]+")),

    # Indian mobile: exactly 10 digits starting with 6-9
    # Must come BEFORE generic BANK_ACC so 10-digit mobiles are not swallowed
    ("PHONE_NUMBER", re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")),

    # Bank account: 9-18 digits — intentionally LAST so everything above
    # is matched first and those positions are already occupied
    ("BANK_ACC",     re.compile(r"(?<!\d)\d{9,18}(?!\d)")),
]

# ── Entity → replacement token ────────────────────────────────────────────────
ENTITY_MAP = {
    "PERSON":       "<n>",
    "EMAIL":        "<EMAIL>",
    "PHONE_NUMBER": "<MOBILE>",
    "AADHAR":       "<AADHAAR>",
    "PAN":          "<PAN>",
    "IFSC":         "<IFSC>",
    "BANK_ACC":     "<ACCOUNT>",
    "DP_ACCOUNT":   "<ACCOUNT>",
    "CIF":          "<CIF>",
    "USER_ID":      "<USER_ID>",
    "TRADING_ID":   "<USER_ID>",
    "DP_ID":        "<DP_ID>",
    "URL":          "<URL>",
    "IP_ADDRESS":   "<IP_ADDR>",
}

# ── Load extra patterns from config.json ──────────────────────────────────────
CONFIG_PATH = r"E:\FeatSystems\masking_data_label_studio\config.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    extra = []
    already_defined = {name for name, _ in REGEX_PATTERNS}
    for item in config.get("patterns", []):
        name  = item["name"]
        regex = item["regex"]
        if name in already_defined:
            continue          # skip — already handled above
        try:
            extra.append((name, re.compile(regex)))
        except re.error as err:
            print(f"  WARNING: bad regex for {name}: {err}")
    # Insert extras before BANK_ACC (last entry) so they get priority
    REGEX_PATTERNS = REGEX_PATTERNS[:-1] + extra + [REGEX_PATTERNS[-1]]
    print(f"Loaded {len(extra)} extra patterns from config")
else:
    print(f"WARNING: config not found at {CONFIG_PATH} — using built-in patterns only")


# ── HTML stripper ─────────────────────────────────────────────────────────────
def strip_html(html_text):
    plain = re.sub(r'<[^>]+>', ' ', html_text)
    plain = re.sub(r'&nbsp;', ' ', plain)
    plain = re.sub(r'&amp;',  '&', plain)
    plain = re.sub(r'&lt;',   '<', plain)
    plain = re.sub(r'&gt;',   '>', plain)
    plain = re.sub(r'\s+',    ' ', plain).strip()
    return plain


# ── Span detection ────────────────────────────────────────────────────────────
def detect_entities(text):
    """
    Returns sorted list of (start, end, entity_type) found in text.

    Strategy:
      1. Run each regex pattern in order (specific → broad).
         A character position can only be claimed once — first match wins.
      2. Run IndicNER for names; only add spans not already occupied.
    """
    spans    = []
    occupied = set()   # claimed character indices

    def is_free(s, e):
        return not occupied.intersection(range(s, e))

    def claim(s, e):
        occupied.update(range(s, e))

    # Regex pass
    for entity_type, pattern in REGEX_PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if is_free(s, e):
                spans.append((s, e, entity_type))
                claim(s, e)

    # IndicNER pass — names only
    try:
        predictions = ner_pipeline(text)
        for p in predictions:
            if p["entity_group"] == "PER":
                s, e = p["start"], p["end"]
                if is_free(s, e):
                    spans.append((s, e, "PERSON"))
                    claim(s, e)
    except Exception as ex:
        print(f"  IndicNER warning: {ex}")

    return sorted(spans, key=lambda x: x[0])


# ── Token builder — flat or indexed ──────────────────────────────────────────
def build_token(entity_type, counts):
    """
    Returns replacement token.
    counts is mutated in-place to track how many times each type has appeared.
    """
    base = ENTITY_MAP.get(entity_type, f"<{entity_type}>")
    if not INDEXED_TOKENS:
        return base
    counts[entity_type] = counts.get(entity_type, 0) + 1
    return base[:-1] + f"_{counts[entity_type]}>"   # <ACCOUNT> → <ACCOUNT_1>


# ── CORE MASKING FUNCTION ─────────────────────────────────────────────────────
def mask_text(text):
    """
    Replace all detected PII in text with tokens.

    Handles plain text and HTML. HTML tags are preserved.

    INDEXED_TOKENS=False  →  <ACCOUNT>, <ACCOUNT>
    INDEXED_TOKENS=True   →  <ACCOUNT_1>, <ACCOUNT_2>
    """
    if not text or not text.strip():
        return text

    is_html = bool(re.search(r'<[a-zA-Z]', text))
    plain   = strip_html(text) if is_html else text

    if not plain.strip():
        return text

    spans = detect_entities(plain)
    if not spans:
        return text

    # Build token plan left-to-right (correct index order)
    counts     = {}
    token_plan = [(s, e, build_token(et, counts)) for s, e, et in spans]

    # Replace right-to-left (positions stay valid)
    masked = plain
    for start, end, token in sorted(token_plan, key=lambda x: x[0], reverse=True):
        masked = masked[:start] + token + masked[end:]

    # For HTML: apply same substitutions to original HTML string
    if is_html:
        result = text
        for start, end, token in sorted(token_plan, key=lambda x: x[0], reverse=True):
            original_word = plain[start:end]
            result = result.replace(original_word, token, 1)
        return result

    return masked


# ── QUICK TEST ────────────────────────────────────────────────────────────────
def run_test():
    test_cases = [
        ("11-digit account",
         "I want to unblock my account number 12345678901 and my phone is 9892333333"),

        ("Two accounts flat",
         "Transfer from account 123456789012 to account 987654321098"),

        ("Three accounts",
         "Accounts 111222333444, 555666777888 and 999000111222 are all blocked"),

        ("CIF + IFSC",
         "My CIF is 12345678 and IFSC code is HDFC0001234"),

        ("Email + account",
         "Update email to user@example.com for account 987654321012"),

        ("IP address",
         "IP access from 192.168.1.45 — please check"),

        ("PAN + Aadhaar",
         "PAN number ABCDE1234F and Aadhaar 1234 5678 9012"),

        ("Name",
         "Dear Rajesh Kumar, your account has been blocked"),

        ("Phone + account",
         "Call me on 9876543210 regarding my account 123456789012"),

        ("URL",
         "Please visit https://bank.example.com/reset to reset"),

        ("DP Account CDSL",
         "DP account 12345678 87654321 needs unblocking"),
    ]

    print("\n" + "=" * 65)
    print(f"MASKING TEST  (INDEXED_TOKENS={INDEXED_TOKENS})")
    print("=" * 65)
    for label, text in test_cases:
        masked = mask_text(text)
        print(f"\n  [{label}]")
        print(f"  IN : {text}")
        print(f"  OUT: {masked}")
    print("=" * 65 + "\n")


# ── LABEL STUDIO INTEGRATION ──────────────────────────────────────────────────
API_TOKEN = ""           # ← fill this
BASE_URL  = "http://localhost:8080/api"
PROXIES   = {"http": None}


def get_tasks(project_id, page=1):
    url = f"{BASE_URL}/tasks/?page={page}&project={project_id}&fields=task_only"
    headers = {
        "accept":        "application/json",
        "Authorization": f"Token {API_TOKEN}"
    }
    try:
        resp = requests.get(url, headers=headers, proxies=PROXIES, verify=False)
        resp.raise_for_status()
        data = resp.json()
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

    url = f"{BASE_URL}/tasks/{task_id}/"
    headers = {
        "accept":        "application/json",
        "Content-Type":  "application/json",
        "Authorization": f"Token {API_TOKEN}"
    }
    resp = requests.patch(
        url,
        headers=headers,
        proxies=PROXIES,
        json={"data": task["data"], "project": task["project"]},
        verify=False
    )

    if resp.status_code in [200, 201]:
        print(f"  Task {task_id}: masked and updated ✓")
    else:
        print(f"  Task {task_id}: FAILED {resp.status_code} — {resp.text[:100]}")


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


# ── INFERENCE IMPORT NOTE ─────────────────────────────────────────────────────
# In Inference_Single_Level.py:
#
#   from masking import mask_text
#   text = mask_text(str(row[EMAIL_COL]))

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_test()

    PROJECT_ID = 16
    process_project(PROJECT_ID)
