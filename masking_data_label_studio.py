import re
import requests
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import Pattern, PatternRecognizer, AnalyzerEngine, RecognizerRegistry, EntityRecognizer, RecognizerResult
import json
import os

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""

import ssl
ssl._create_default_https_context = ssl.SSLContext

# =========================
# GPU setup
# =========================
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

# =========================
# Load IndicNER for PERSON
# =========================
model_name = r"E:\FeatSystems\masking_data_label_studio\indicmodel"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)


class IndicNERRecognizer(EntityRecognizer):
    def __init__(self, ner_pipeline, supported_entities=None, name="IndicNERRecognizer"):
        self.ner_pipeline = ner_pipeline
        supported_entities = supported_entities or ["PERSON"]
        super().__init__(supported_entities=supported_entities, name=name)

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        predictions = self.ner_pipeline(text)
        for p in predictions:
            if p["entity_group"] == "PER" and "PERSON" in entities:
                results.append(
                    RecognizerResult(
                        entity_type="PERSON",
                        start=p["start"],
                        end=p["end"],
                        score=p["score"]
                    )
                )
        return results


# =========================
# Regex Patterns from Config
# =========================
CONFIG_PATH = r"E:\FeatSystems\masking_data_label_studio\config.json"
patterns = []
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    patterns = [(item["name"], item["regex"]) for item in config.get("patterns", [])]
else:
    print(f"Config file not found at {CONFIG_PATH}")

# =========================
# Entity → Tag label mapping
# =========================
# Maps entity type names to the display tag used in masked output.
# Add or edit entries here if you rename entities in config.json.
ENTITY_TAG_MAP = {
    "Account_number":   "ACCOUNT_NUMBER",
    "DP_ID":            "DP_ID",
    "TRADING_ID":       "TRADING_ID",
    "PAN":              "PAN",
    "DP_ACCOUNT_CDSL":  "DP_ACCOUNT_CDSL",
    "BANK_ACC":         "BANK_ACCOUNT",
    "AADHAR":           "AADHAR",
    "IFSC":             "IFSC",
    "PHONE_NUMBER":     "PHONE_NUMBER",
    "EMAIL_ADDRESS":    "EMAIL",
    "URL":              "URL",
    "PERSON":           "NAME",
}


def get_tag(entity_type, counter=None):
    """Return a bracketed NER-style tag for the given entity type."""
    label = ENTITY_TAG_MAP.get(entity_type, entity_type)
    if counter is not None:
        return f"[{label}_{counter}]"
    return f"[{label}]"


# =========================
# Register recognizers
# =========================
registry = RecognizerRegistry()
registry.load_predefined_recognizers()   # includes PHONE_NUMBER, EMAIL_ADDRESS

for name, regex in patterns:
    pattern = Pattern(name=f"{name} pattern", regex=regex, score=1.0)
    recognizer = PatternRecognizer(supported_entity=name, patterns=[pattern])
    registry.add_recognizer(recognizer)

registry.add_recognizer(IndicNERRecognizer(ner_pipeline))

analyzer = AnalyzerEngine(registry=registry)


# =========================
# Masking
# =========================
def mask_text(text):
    entities_to_mask = [e[0] for e in patterns] + ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]
    results = analyzer.analyze(text=text, entities=entities_to_mask, language='en')

    # Sort in reverse order to avoid offset shifting issues
    results = sorted(results, key=lambda x: x.start, reverse=True)
    masked_text = text

    # Per-entity-type counters so each occurrence gets a unique numbered tag
    # e.g. [NAME_1], [NAME_2] ... [EMAIL_1], [EMAIL_2] ...
    counters = {}

    for r in results:
        start, end = r.start, r.end
        original = masked_text[start:end]

        # Skip URLs that start with www. (likely not sensitive)
        if original.strip().lower().startswith("www."):
            continue

        entity_type = r.entity_type

        if entity_type == "PERSON":
            # Expand boundaries to capture full name tokens
            while start > 0 and text[start - 1].isalpha():
                start -= 1
            while end < len(text) and text[end].isalpha():
                end += 1

        counters[entity_type] = counters.get(entity_type, 0) + 1
        tag = get_tag(entity_type, counters[entity_type])

        if entity_type == "PERSON":
            # Keep a trailing space to separate from following text
            masked_text = masked_text[:start] + tag + " " + masked_text[end:]
        else:
            masked_text = masked_text[:start] + tag + masked_text[end:]

    # Merge consecutive NAME tags into a single one (e.g. [NAME_1] [NAME_2] → [NAME_1])
    masked_text = re.sub(
        r'\[NAME_\d+\](\s*\[NAME_\d+\])+',
        lambda m: m.group(0).split(']')[0] + ']',
        masked_text
    )

    return masked_text


# =========================
# Label Studio API
# =========================
API_TOKEN = ""
BASE_URL = "http://localhost:8080/api"
PROXIES = {"http": None}


def get_tasks(project_id, page=1):
    url = f"{BASE_URL}/tasks/?page={page}&project={project_id}&fields=task_only"
    headers = {
        "accept": "application/json",
        "Authorization": f"Token {API_TOKEN}"
    }
    try:
        response = requests.get(url, headers=headers, proxies=PROXIES, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching page {page}: {e}")
        return []

    data = response.json()
    if isinstance(data, dict):
        return data.get("results", []) or data.get("tasks", [])
    return data


def update_task(task):
    task_id = task["id"]
    original_html = task.get("data", {}).get("html", "")
    masked_html = mask_text(original_html)
    task["data"]["html"] = masked_html

    url = f"{BASE_URL}/tasks/{task_id}/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Token {API_TOKEN}"
    }
    response = requests.patch(
        url,
        headers=headers,
        proxies=PROXIES,
        json={"data": task["data"], "project": task["project"]}
    )

    if response.status_code in [200, 201]:
        print(f"✅ Task {task_id} updated successfully.")
    else:
        print(f"❌ Failed to update task {task_id}: {response.status_code} - {response.text}")


def process_project_tasks(project_id):
    page = 1
    total_processed = 0

    while True:
        tasks = get_tasks(project_id, page=page)
        if not tasks:
            print(f"✅ No more tasks found. Total processed: {total_processed}")
            break

        print(f"📄 Processing page {page} with {len(tasks)} tasks...")
        for task in tasks:
            update_task(task)
            total_processed += 1

        page += 1


# =========================
# Run
# =========================
if __name__ == "__main__":
    PROJECT_ID = 16
    process_project_tasks(PROJECT_ID)
