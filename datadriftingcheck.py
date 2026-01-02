


import pandas as pd
import json
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# =====================================================
# CONFIG
# =====================================================
REFERENCE_PATH = r"C:\SM_AJIT\Jupyter\DataDrifting\ReferenceFileForDataDrifting.csv"
CURRENT_PATH   = r"C:\SM_AJIT\Jupyter\DataDrifting\CurrentFileForDataDrifting.csv"

CATEGORY_VOLUME_THRESHOLD = 0.10   # 10%
AUGMENT_SIM_THRESHOLD = 0.75       # cosine similarity

# =====================================================
# 1. LOAD DATA
# =====================================================
reference_df = pd.read_csv(REFERENCE_PATH)
current_df   = pd.read_csv(CURRENT_PATH)

required_cols = ["Subject", "classification", "html", "augmented_html"]
reference_df = reference_df[required_cols].dropna()
current_df   = current_df[required_cols].dropna()

print("Reference rows:", len(reference_df))
print("Current rows:", len(current_df))

# =====================================================
# 2. FLATTEN TAXONOMY LABEL
# =====================================================
def flatten_taxonomy(classification_str):
    try:
        data = json.loads(classification_str)
        path = data[0]["taxonomy"][0]
        return " > ".join(path)
    except:
        return None

reference_df["classification_flat"] = reference_df["classification"].apply(flatten_taxonomy)
current_df["classification_flat"]   = current_df["classification"].apply(flatten_taxonomy)

reference_df.dropna(subset=["classification_flat"], inplace=True)
current_df.dropna(subset=["classification_flat"], inplace=True)

# =====================================================
# 3. CONCEPT DRIFT (NEW / REMOVED CATEGORIES)
# =====================================================
old_categories = set(reference_df["classification_flat"].unique())
new_categories = set(current_df["classification_flat"].unique())

added_categories   = new_categories - old_categories
removed_categories = old_categories - new_categories

print("\n🔴 CONCEPT DRIFT CHECK")
print("New categories:", len(added_categories))
print("Removed categories:", len(removed_categories))

for c in sorted(added_categories):
    print(" +", c)

# =====================================================
# 4. CATEGORY VOLUME DRIFT (PER CATEGORY >10%)
# =====================================================
ref_counts = Counter(reference_df["classification_flat"])
cur_counts = Counter(current_df["classification_flat"])

volume_drift_rows = []

for cat in set(ref_counts) | set(cur_counts):
    ref = ref_counts.get(cat, 0)
    cur = cur_counts.get(cat, 0)

    if ref == 0:
        change_pct = 1.0
    else:
        change_pct = (cur - ref) / ref

    if abs(change_pct) >= CATEGORY_VOLUME_THRESHOLD:
        volume_drift_rows.append({
            "category": cat,
            "reference_count": ref,
            "current_count": cur,
            "change_pct": round(change_pct * 100, 2)
        })

volume_drift_df = pd.DataFrame(volume_drift_rows)

print("\n🟠 CATEGORY VOLUME DRIFT CHECK (>10%)")
if volume_drift_df.empty:
    print("✅ No significant category volume drift")
else:
    print(volume_drift_df.sort_values("change_pct", ascending=False))

# =====================================================
# 5. DATA DRIFT DASHBOARD (EVIDENTLY)
# =====================================================
data_report = Report(metrics=[DataDriftPreset()])

data_report.run(
    reference_data=reference_df[["Subject", "html"]],
    current_data=current_df[["Subject", "html"]],
)

dashboard_name = f"data_drift_dashboard_{datetime.now().date()}.html"
data_report.save_html(dashboard_name)

print("\n🟡 DATA DRIFT DASHBOARD")
print("📊 Saved as:", dashboard_name)

# =====================================================
# 6. AUGMENTATION DRIFT (TEXT QUALITY CHECK)
# =====================================================
texts_original = reference_df["html"].astype(str).tolist()
texts_augmented = reference_df["augmented_html"].astype(str).tolist()

vectorizer = TfidfVectorizer(min_df=3, max_features=5000)
tfidf = vectorizer.fit_transform(texts_original + texts_augmented)

orig_vecs = tfidf[:len(texts_original)]
aug_vecs  = tfidf[len(texts_original):]

similarities = cosine_similarity(orig_vecs, aug_vecs).diagonal()

low_quality_count = (similarities < AUGMENT_SIM_THRESHOLD).sum()
augmentation_drift_ratio = low_quality_count / len(similarities)

print("\n🔥 AUGMENTATION DRIFT CHECK")
print("Low similarity rows (<0.75):", low_quality_count)
print("Augmentation drift ratio:", round(augmentation_drift_ratio * 100, 2), "%")

# =====================================================
# 7. FINAL DECISION ENGINE
# =====================================================
print("\n🚦 FINAL DECISION")

if added_categories:
    print("🔴 CONCEPT DRIFT → RETRAIN MODEL (MANDATORY)")
elif not volume_drift_df.empty:
    print("🟠 CATEGORY VOLUME DRIFT → MONITOR / PARTIAL RETRAIN")
elif augmentation_drift_ratio > 0.3:
    print("🟠 AUGMENTATION DRIFT → FIX DATA AUGMENTATION")
else:
    print("🟢 SYSTEM STABLE")

print("\nDone.")
