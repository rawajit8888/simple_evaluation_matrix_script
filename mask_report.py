import re
import torch
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import (
    Pattern, PatternRecognizer, AnalyzerEngine,
    RecognizerRegistry, EntityRecognizer, RecognizerResult
)

# =========================
# Configuration — EDIT THESE
# =========================
INPUT_FILE_PATH  = r"E:\Projects\masking_label_studio_data\abc.csv"   # your input file (.csv or .xlsx)
CONFIG_PATH      = r"E:\Projects\masking_label_studio_data\config.json"
MODEL_PATH       = r"E:\Projects\masking_label_studio_data\indi_ner_model"
INPUT_COLUMN     = "data"   # column name in your CSV

# Output files — saved in same folder as input automatically
_input_dir       = os.path.dirname(INPUT_FILE_PATH)
OUTPUT_CSV_PATH  = os.path.join(_input_dir, "masked_output.csv")        # masked data
OUTPUT_REPORT_PATH = os.path.join(_input_dir, "detection_report.xlsx")  # detection report

# =========================
# GPU setup
# =========================
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

# =========================
# Load IndicNER for PERSON
# =========================
print("Loading NER model...")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
model        = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

class IndicNERRecognizer(EntityRecognizer):
    def __init__(self, ner_pipeline, supported_entities=None, name="IndicNERRecognizer"):
        self.ner_pipeline  = ner_pipeline
        supported_entities = supported_entities or ["PERSON"]
        super().__init__(supported_entities=supported_entities, name=name)

    def analyze(self, text, entities, nlp_artifacts=None):
        results     = []
        predictions = self.ner_pipeline(text)
        for p in predictions:
            if p["entity_group"] == "PER" and "PERSON" in entities:
                results.append(RecognizerResult(
                    entity_type="PERSON",
                    start=p["start"],
                    end=p["end"],
                    score=p["score"]
                ))
        return results

# =========================
# Regex Patterns from Config
# =========================
patterns = []
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    patterns = [(item["name"], item["regex"]) for item in config.get("patterns", [])]
    print(f"✅ Loaded {len(patterns)} regex patterns from config.")
else:
    print(f"⚠️  Config file not found at {CONFIG_PATH}, continuing without custom patterns.")

# =========================
# Register recognizers
# =========================
registry = RecognizerRegistry()
registry.load_predefined_recognizers()  # includes PHONE_NUMBER, EMAIL_ADDRESS

for name, regex in patterns:
    pattern    = Pattern(name=f"{name} pattern", regex=regex, score=1.0)
    recognizer = PatternRecognizer(supported_entity=name, patterns=[pattern])
    registry.add_recognizer(recognizer)

registry.add_recognizer(IndicNERRecognizer(ner_pipeline))
analyzer = AnalyzerEngine(registry=registry)
print("✅ Analyzer engine ready.\n")

# =========================
# Masking helpers
# =========================
def partial_mask(value, unmasked_digits=4):
    """Mask all but last few characters."""
    if len(value) <= unmasked_digits:
        return "X" * len(value)
    return "X" * (len(value) - unmasked_digits) + value[-unmasked_digits:]


def mask_and_detect(text):
    """
    Masks PII in text and returns:
        masked_text   : str  — masked version
        detected      : list — one dict per detected entity
    """
    if not isinstance(text, str) or not text.strip():
        return text, []

    entities_to_mask = [e[0] for e in patterns] + ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]
    results          = analyzer.analyze(text=text, entities=entities_to_mask, language='en')
    results          = sorted(results, key=lambda x: x.start, reverse=True)

    masked_text = text
    email_count = 0
    name_count  = 0
    detected    = []

    for r in results:
        start, end = r.start, r.end
        original   = masked_text[start:end]

        # Skip www. tokens
        if original.strip().lower().startswith("www."):
            continue

        if r.entity_type == "PERSON":
            s = r.start
            while s > 0 and text[s - 1].isalpha():
                s -= 1
            e = r.end
            while e < len(text) and text[e].isalpha():
                e += 1
            full_name   = text[s:e]
            name_count += 1
            replacement = f"[NAME {name_count}]"
            masked_text = masked_text[:s] + replacement + " " + masked_text[e:]
            detected.append({
                "entity_type"   : "PERSON",
                "original_value": full_name,
                "masked_as"     : replacement,
                "confidence"    : round(r.score, 4),
                "start"         : s,
                "end"           : e
            })

        elif r.entity_type == "EMAIL_ADDRESS":
            email_count += 1
            replacement  = f"[EMAIL {email_count}]"
            masked_text  = masked_text[:start] + replacement + masked_text[end:]
            detected.append({
                "entity_type"   : "EMAIL_ADDRESS",
                "original_value": original,
                "masked_as"     : replacement,
                "confidence"    : round(r.score, 4),
                "start"         : start,
                "end"           : end
            })

        elif r.entity_type == "URL":
            masked_text = masked_text[:start] + "[URL]" + masked_text[end:]
            detected.append({
                "entity_type"   : "URL",
                "original_value": original,
                "masked_as"     : "[URL]",
                "confidence"    : round(r.score, 4),
                "start"         : start,
                "end"           : end
            })

        elif r.entity_type in [
            "DP_ACCOUNT", "DP_ID", "TRADING_ID", "PAN",
            "DP_ACCOUNT_CDSL", "BANK_ACC", "AADHAR", "PHONE_NUMBER", "IFSC"
        ]:
            replacement = partial_mask(original)
            masked_text = masked_text[:start] + replacement + masked_text[end:]
            detected.append({
                "entity_type"   : r.entity_type,
                "original_value": original,
                "masked_as"     : replacement,
                "confidence"    : round(r.score, 4),
                "start"         : start,
                "end"           : end
            })

    # Collapse consecutive [NAME X] tags into one
    masked_text = re.sub(
        r'\[NAME \d+\](\s*\[NAME \d+\])+',
        lambda m: m.group(0).split(']')[0] + ']',
        masked_text
    )

    return masked_text, detected


# =========================
# Load file helper
# =========================
def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .xlsx, .xls, or .csv")
    df.columns = df.columns.str.strip()  # strip accidental spaces from headers
    return df


# =========================
# Main
# =========================
if __name__ == "__main__":

    # ── Load input ──────────────────────────────────────────
    print(f"📂 Loading: {INPUT_FILE_PATH}")
    df = load_file(INPUT_FILE_PATH)

    if INPUT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{INPUT_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    total = len(df)
    print(f"🔢 Total rows to process: {total}\n")

    # ── Process each row ────────────────────────────────────
    masked_values = []
    row_report    = []
    detail_report = []

    for i, text in enumerate(df[INPUT_COLUMN], start=1):
        masked, detected = mask_and_detect(text)
        masked_values.append(masked)

        # Row-level summary
        entity_types   = ", ".join(sorted(set(d["entity_type"] for d in detected))) if detected else "NONE"
        entity_count   = len(detected)
        avg_confidence = round(sum(d["confidence"] for d in detected) / entity_count, 4) if detected else None

        row_report.append({
            "row_number"    : i,
            "original_text" : text,
            "masked_text"   : masked,
            "entities_found": entity_types,
            "total_entities": entity_count,
            "avg_confidence": avg_confidence,
            "has_pii"       : "YES" if detected else "NO"
        })

        # Entity-level details
        for d in detected:
            detail_report.append({
                "row_number"    : i,
                "original_text" : text,
                "entity_type"   : d["entity_type"],
                "original_value": d["original_value"],
                "masked_as"     : d["masked_as"],
                "confidence"    : d["confidence"],
                "start_pos"     : d["start"],
                "end_pos"       : d["end"]
            })

        if i % 50 == 0 or i == total:
            print(f"   ✔ Processed {i}/{total} rows")

    # ── Save masked_output.csv ──────────────────────────────
    input_col_idx = df.columns.get_loc(INPUT_COLUMN)
    df.insert(input_col_idx + 1, "masked_data", masked_values)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ Masked CSV saved  → {OUTPUT_CSV_PATH}")

    # ── Build report dataframes ─────────────────────────────
    df_rows    = pd.DataFrame(row_report)
    df_details = pd.DataFrame(detail_report) if detail_report else pd.DataFrame(
        columns=["row_number", "original_text", "entity_type",
                 "original_value", "masked_as", "confidence", "start_pos", "end_pos"]
    )

    # Entity summary
    if detail_report:
        summary = (
            df_details.groupby("entity_type")
            .agg(
                total_detected=("entity_type", "count"),
                avg_confidence=("confidence", "mean"),
                min_confidence=("confidence", "min"),
                max_confidence=("confidence", "max")
            )
            .reset_index()
            .sort_values("total_detected", ascending=False)
        )
        for col in ["avg_confidence", "min_confidence", "max_confidence"]:
            summary[col] = summary[col].round(4)
    else:
        summary = pd.DataFrame(columns=["entity_type", "total_detected",
                                        "avg_confidence", "min_confidence", "max_confidence"])

    # Overall stats
    rows_with_pii    = int((df_rows["has_pii"] == "YES").sum())
    rows_without_pii = total - rows_with_pii
    total_entities   = len(df_details)

    stats = pd.DataFrame([
        {"metric": "Total Rows Processed",     "value": total},
        {"metric": "Rows WITH PII detected",   "value": rows_with_pii},
        {"metric": "Rows WITHOUT PII",         "value": rows_without_pii},
        {"metric": "PII Detection Rate (%)",   "value": round(rows_with_pii / total * 100, 2) if total else 0},
        {"metric": "Total Entities Detected",  "value": total_entities},
        {"metric": "Avg Entities per PII Row", "value": round(total_entities / rows_with_pii, 2) if rows_with_pii else 0},
    ])

    # ── Save detection_report.xlsx ──────────────────────────
    with pd.ExcelWriter(OUTPUT_REPORT_PATH, engine="openpyxl") as writer:
        df_rows.to_excel(writer,    sheet_name="Row Report",     index=False)
        df_details.to_excel(writer, sheet_name="Entity Details", index=False)
        summary.to_excel(writer,    sheet_name="Entity Summary", index=False)
        stats.to_excel(writer,      sheet_name="Overall Stats",  index=False)

    print(f"✅ Detection report saved → {OUTPUT_REPORT_PATH}")

    # ── Quick summary in terminal ───────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊  SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total rows processed  : {total}
  Rows WITH PII         : {rows_with_pii}  ({round(rows_with_pii/total*100, 1) if total else 0}%)
  Rows WITHOUT PII      : {rows_without_pii}
  Total entities found  : {total_entities}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Output files:
  📄 {OUTPUT_CSV_PATH}
  📊 {OUTPUT_REPORT_PATH}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)