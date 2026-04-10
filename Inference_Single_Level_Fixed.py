import requests
import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

PREDICT_URL_BERT = "http://localhost:9093/predictext"

PROXIES = {
    "http": None
}

CSV_PATH = r"E:\FeatSystems\Test_environment\single_level_test_data_900.csv"

EMAIL_COL = "Email Message"
TRUE_COL  = "classification_true"
PRED_COL  = "classification_pred"
CONF_COL  = "confidence_score"

# =========================
# PAYLOAD BUILDER — FIXED
# json.dumps() removed — was corrupting email text
# =========================
def build_payload_bert(text):
    return {
        "texts": [
            {
                "id": "3",
                "text": text          # ← FIXED: raw string, no json.dumps()
            }
        ]
    }

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH, encoding="utf-8", engine="python")

if PRED_COL not in df.columns:
    df[PRED_COL] = ""

if CONF_COL not in df.columns:
    df[CONF_COL] = ""

print(f"Loaded {len(df)} rows")

headers = {
    "Content-Type": "application/json"
}

# =========================
# BATCH PREDICTION LOOP
# =========================
no_match_count = 0
error_count     = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):

    text = str(row[EMAIL_COL])

    if not text.strip():
        df.at[idx, PRED_COL] = "EMPTY"
        df.at[idx, CONF_COL] = 0.0
        continue

    payload = build_payload_bert(text)

    try:
        response = requests.post(
            PREDICT_URL_BERT,
            headers=headers,
            json=payload,
            proxies=PROXIES,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        # ── Safe parse with guards ──────────────────────────────
        results_list = result.get("results", [])
        if not results_list:
            print(f"Row {idx}: empty results from API")
            df.at[idx, PRED_COL] = "NO_RESULT"
            df.at[idx, CONF_COL] = 0.0
            no_match_count += 1
            continue

        api_block = results_list[0].get("result", [])

        classification_pred = ""
        confidence_score    = 0.0

        for item in api_block:
            if item.get("from_name") == "classification":
                taxonomy_path       = item["value"]["taxonomy"][0]
                classification_pred = " > ".join(taxonomy_path)
                confidence_score    = item["value"].get("score", 0.0)
                break

        # ── Check if classification was found ───────────────────
        if not classification_pred:
            print(f"Row {idx}: 'classification' from_name not found in response")
            df.at[idx, PRED_COL] = "NO_MATCH"
            df.at[idx, CONF_COL] = 0.0
            no_match_count += 1
            continue

        df.at[idx, PRED_COL] = classification_pred
        df.at[idx, CONF_COL] = round(float(confidence_score), 4)

    except Exception as e:
        print(f"Row {idx} failed: {e}")
        df.at[idx, PRED_COL] = "ERROR"
        df.at[idx, CONF_COL] = 0.0
        error_count += 1

print(f"\nDone. Errors: {error_count} | No match: {no_match_count}")

# =========================
# SAVE PREDICTIONS
# =========================
OUT_PATH = "test_single_level_test_data_900_09_04_2026.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Saved predictions → {OUT_PATH}")

# =========================
# EVALUATION METRICS
# =========================
print("\nRunning evaluation...")

eval_df = df[~df[PRED_COL].isin(["ERROR", "NO_RESULT", "NO_MATCH", "EMPTY"])].copy()

print(f"Evaluating on {len(eval_df)} rows (excluded {len(df) - len(eval_df)} error rows)")

y_true = eval_df[TRUE_COL]
y_pred = eval_df[PRED_COL]

acc       = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n===== OVERALL METRICS =====")
print(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

report = classification_report(y_true, y_pred, zero_division=0)
print("\n===== CLASSIFICATION REPORT =====")
print(report)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
print("\n===== CONFUSION MATRIX =====")
print(cm)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.show()

# =========================
# SAVE METRICS FILE
# =========================
METRICS_PATH = "metrics_single_level_900_09_04_2026.txt"
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write(f"Total rows     : {len(df)}\n")
    f.write(f"Evaluated rows : {len(eval_df)}\n")
    f.write(f"Error rows     : {error_count}\n")
    f.write(f"No match rows  : {no_match_count}\n\n")
    f.write(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n\n")
    f.write(report)

print(f"Saved metrics → {METRICS_PATH}")
