import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# ================= CONFIG =================
EXCEL_PATH = r"E:\Projects\Interactive AI\label-studio-ml-backend\results\bert-classification-sentiment\evaluation\test_predictions.csv"   # <-- change path
        # <-- your csv file
TRUE_COL = "classification_true"
PRED_COL = "classification_pred"
OUTPUT_FILE = "metrics.txt"
# =========================================


def evaluate_classification_from_excel():
    df = pd.read_csv(EXCEL_PATH)

    if TRUE_COL not in df.columns or PRED_COL not in df.columns:
        raise ValueError("Required columns not found in Excel")

    y_true = df[TRUE_COL].astype(str).tolist()
    y_pred = df[PRED_COL].astype(str).tolist()

    labels = sorted(set(y_true))

    lines = []  # store output for file

    def log(text=""):
        print(text)
        lines.append(text)

    log("\n=== Classification Metrics ===")

    overall_accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    log(f"Accuracy: {overall_accuracy:.4f}")
    log(f"F1 (weighted): {weighted_f1:.4f}\n")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = np.sum(cm)

    log("Classification report:\n")
    header = (
        f"{'':70s}"
        f"{'precision':>10s}"
        f"{'recall':>10s}"
        f"{'f1-score':>10s}"
        f"{'accuracy':>10s}"
        f"{'support':>10s}"
    )
    log(header)

    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = total - (TP + FP + FN)

        class_accuracy = (TP + TN) / total

        row = (
            f"{label:70s}"
            f"{report[label]['precision']:10.2f}"
            f"{report[label]['recall']:10.2f}"
            f"{report[label]['f1-score']:10.2f}"
            f"{class_accuracy:10.2f}"
            f"{int(report[label]['support']):10d}"
        )
        log(row)

    total_support = sum(report[l]["support"] for l in labels)

    log("\n" + "macro avg".ljust(70) +
        f"{report['macro avg']['precision']:10.2f}"
        f"{report['macro avg']['recall']:10.2f}"
        f"{report['macro avg']['f1-score']:10.2f}"
        f"{overall_accuracy:10.2f}"
        f"{int(total_support):10d}"
    )

    log("weighted avg".ljust(70) +
        f"{report['weighted avg']['precision']:10.2f}"
        f"{report['weighted avg']['recall']:10.2f}"
        f"{report['weighted avg']['f1-score']:10.2f}"
        f"{overall_accuracy:10.2f}"
        f"{int(total_support):10d}"
    )

    # ===== SAVE TO FILE =====
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Metrics saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_classification_from_excel()

