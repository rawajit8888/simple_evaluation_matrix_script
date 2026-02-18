import pandas as pd
import requests
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
AUTH_URL = "http://10.176.3.178:5080/api/Auth/token"
PREDICT_URL_BERT = "http://10.176.3.178:5080/api/External/Bert-Multi-Task-Classifier/predict"

CSV_PATH = "emails.csv"   # <-- change to your actual path

# ===== CSV COLUMNS =====
EMAIL_COL = "email"                                  # column with email text

# Ground truth columns (if you have them)
TRUE_MASTERDEPT_COL = "masterdepartment_true"       # e.g., "Internet Banking"
TRUE_DEPT_COL       = "department_true"              # e.g., "Internet Banking > Account Access"
TRUE_QUERYTYPE_COL  = "querytype_true"               # e.g., "Internet Banking > Account Access > Unblock"

# Prediction columns (will be created)
PRED_MASTERDEPT_COL = "masterdepartment_pred"
PRED_DEPT_COL       = "department_pred"
PRED_QUERYTYPE_COL  = "querytype_pred"

# Confidence columns (will be created)
CONF_MASTERDEPT_COL = "masterdepartment_confidence"
CONF_DEPT_COL       = "department_confidence"
CONF_QUERYTYPE_COL  = "querytype_confidence"


# =========================
# AUTH
# =========================
def get_token():
    """Get authentication token"""
    credentials = {
        "clientId": "client1_id",
        "clientSecret": "client1_secret"
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = requests.post(AUTH_URL, headers=headers, json=credentials)
        resp.raise_for_status()
        return resp.json().get("token")
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        raise


# =========================
# PAYLOAD BUILDER
# =========================
def build_payload(text):
    """Build API request payload"""
    return {
        "texts": [
            {
                "id": "1",
                "text": text
            }
        ]
    }


# =========================
# PARSE 3-LEVEL RESPONSE
# =========================
def parse_3level_response(result):
    """
    Parse API response to extract all 3 levels + confidence scores
    
    Returns:
        dict with keys: masterdepartment, department, querytype,
                        conf_masterdept, conf_dept, conf_querytype
    """
    parsed = {
        'masterdepartment': '',
        'department': '',
        'querytype': '',
        'conf_masterdept': '',
        'conf_dept': '',
        'conf_querytype': ''
    }
    
    try:
        api_block = result["results"][0]["result"]
        
        for item in api_block:
            from_name = item.get("from_name", "")
            taxonomy_path = item.get("value", {}).get("taxonomy", [[]])[0]
            score = item.get("value", {}).get("score", 0)
            
            if from_name == "masterdepartment":
                # Level 1: Just the master department name
                parsed['masterdepartment'] = taxonomy_path[0] if taxonomy_path else ""
                parsed['conf_masterdept'] = score
                
            elif from_name == "department":
                # Level 2: Full path like "Internet Banking > Account Access"
                parsed['department'] = " > ".join(taxonomy_path)
                parsed['conf_dept'] = score
                
            elif from_name == "querytype":
                # Level 3: Full path like "Internet Banking > Account Access > Unblock"
                parsed['querytype'] = " > ".join(taxonomy_path)
                parsed['conf_querytype'] = score
                
    except Exception as e:
        print(f"⚠️  Parse error: {e}")
    
    return parsed


# =========================
# MAIN SCRIPT
# =========================
def main():
    print("=" * 80)
    print("🚀 3-LEVEL HIERARCHICAL INFERENCE SCRIPT")
    print("=" * 80)
    
    # ===== LOAD CSV =====
    print(f"\n📂 Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} rows")
    
    # Add prediction columns if they don't exist
    for col in [PRED_MASTERDEPT_COL, PRED_DEPT_COL, PRED_QUERYTYPE_COL,
                CONF_MASTERDEPT_COL, CONF_DEPT_COL, CONF_QUERYTYPE_COL]:
        if col not in df.columns:
            df[col] = ""
    
    # ===== GET TOKEN =====
    print("\n🔐 Getting authentication token...")
    token = get_token()
    print("✓ Token obtained")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # ===== BATCH PREDICTION =====
    print(f"\n🔮 Running predictions on {len(df)} emails...")
    print("=" * 80)
    
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        
        text = str(row[EMAIL_COL])
        
        if not text.strip():
            df.at[idx, PRED_MASTERDEPT_COL] = "EMPTY"
            df.at[idx, PRED_DEPT_COL] = "EMPTY"
            df.at[idx, PRED_QUERYTYPE_COL] = "EMPTY"
            continue
        
        payload = build_payload(text)
        
        try:
            response = requests.post(
                PREDICT_URL_BERT,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse all 3 levels
            parsed = parse_3level_response(result)
            
            # Save to dataframe
            df.at[idx, PRED_MASTERDEPT_COL] = parsed['masterdepartment']
            df.at[idx, PRED_DEPT_COL]       = parsed['department']
            df.at[idx, PRED_QUERYTYPE_COL]  = parsed['querytype']
            
            df.at[idx, CONF_MASTERDEPT_COL] = parsed['conf_masterdept']
            df.at[idx, CONF_DEPT_COL]       = parsed['conf_dept']
            df.at[idx, CONF_QUERYTYPE_COL]  = parsed['conf_querytype']
            
        except Exception as e:
            error_count += 1
            print(f"\n⚠️  Row {idx} failed: {e}")
            df.at[idx, PRED_MASTERDEPT_COL] = "ERROR"
            df.at[idx, PRED_DEPT_COL]       = "ERROR"
            df.at[idx, PRED_QUERYTYPE_COL]  = "ERROR"
    
    print(f"\n✓ Predictions complete")
    if error_count > 0:
        print(f"⚠️  {error_count} errors occurred")
    
    # ===== SAVE PREDICTIONS =====
    OUT_PATH = "emails_with_3level_predictions.csv"
    df.to_csv(OUT_PATH, index=False)
    print(f"\n💾 Saved predictions → {OUT_PATH}")
    
    # ===== EVALUATION (if ground truth exists) =====
    if TRUE_MASTERDEPT_COL in df.columns and TRUE_DEPT_COL in df.columns and TRUE_QUERYTYPE_COL in df.columns:
        print("\n" + "=" * 80)
        print("📊 EVALUATION METRICS")
        print("=" * 80)
        
        # Filter out errors
        eval_df = df[
            (df[PRED_MASTERDEPT_COL] != "ERROR") & 
            (df[PRED_MASTERDEPT_COL] != "EMPTY")
        ].copy()
        
        print(f"\n✓ Evaluating on {len(eval_df)} valid predictions")
        
        # ===== LEVEL 1: MASTERDEPARTMENT =====
        print("\n" + "-" * 80)
        print("LEVEL 1: MASTERDEPARTMENT")
        print("-" * 80)
        
        y_true_l1 = eval_df[TRUE_MASTERDEPT_COL]
        y_pred_l1 = eval_df[PRED_MASTERDEPT_COL]
        
        acc_l1 = accuracy_score(y_true_l1, y_pred_l1)
        f1_l1  = f1_score(y_true_l1, y_pred_l1, average="weighted", zero_division=0)
        
        print(f"Accuracy: {acc_l1:.4f} ({acc_l1*100:.2f}%)")
        print(f"F1 Score: {f1_l1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true_l1, y_pred_l1, zero_division=0))
        
        # Confusion matrix
        cm_l1 = confusion_matrix(y_true_l1, y_pred_l1)
        
        # ===== LEVEL 2: DEPARTMENT =====
        print("\n" + "-" * 80)
        print("LEVEL 2: DEPARTMENT")
        print("-" * 80)
        
        y_true_l2 = eval_df[TRUE_DEPT_COL]
        y_pred_l2 = eval_df[PRED_DEPT_COL]
        
        acc_l2 = accuracy_score(y_true_l2, y_pred_l2)
        f1_l2  = f1_score(y_true_l2, y_pred_l2, average="weighted", zero_division=0)
        
        print(f"Accuracy: {acc_l2:.4f} ({acc_l2*100:.2f}%)")
        print(f"F1 Score: {f1_l2:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true_l2, y_pred_l2, zero_division=0))
        
        cm_l2 = confusion_matrix(y_true_l2, y_pred_l2)
        
        # ===== LEVEL 3: QUERYTYPE =====
        print("\n" + "-" * 80)
        print("LEVEL 3: QUERYTYPE")
        print("-" * 80)
        
        y_true_l3 = eval_df[TRUE_QUERYTYPE_COL]
        y_pred_l3 = eval_df[PRED_QUERYTYPE_COL]
        
        acc_l3 = accuracy_score(y_true_l3, y_pred_l3)
        f1_l3  = f1_score(y_true_l3, y_pred_l3, average="weighted", zero_division=0)
        
        print(f"Accuracy: {acc_l3:.4f} ({acc_l3*100:.2f}%)")
        print(f"F1 Score: {f1_l3:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true_l3, y_pred_l3, zero_division=0))
        
        cm_l3 = confusion_matrix(y_true_l3, y_pred_l3)
        
        # ===== SUMMARY TABLE =====
        print("\n" + "=" * 80)
        print("📈 SUMMARY - ALL 3 LEVELS")
        print("=" * 80)
        
        summary_df = pd.DataFrame({
            'Level': ['Level 1 (MasterDepartment)', 'Level 2 (Department)', 'Level 3 (QueryType)'],
            'Accuracy': [f"{acc_l1:.4f}", f"{acc_l2:.4f}", f"{acc_l3:.4f}"],
            'F1-Score': [f"{f1_l1:.4f}", f"{f1_l2:.4f}", f"{f1_l3:.4f}"],
            'Accuracy %': [f"{acc_l1*100:.2f}%", f"{acc_l2*100:.2f}%", f"{acc_l3*100:.2f}%"]
        })
        
        print(summary_df.to_string(index=False))
        
        # ===== SAVE METRICS =====
        with open("metrics_3level.txt", "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("3-LEVEL HIERARCHICAL CLASSIFICATION METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("LEVEL 1: MASTERDEPARTMENT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {acc_l1:.4f} ({acc_l1*100:.2f}%)\n")
            f.write(f"F1 Score: {f1_l1:.4f}\n\n")
            f.write(classification_report(y_true_l1, y_pred_l1, zero_division=0))
            f.write("\n\n")
            
            f.write("LEVEL 2: DEPARTMENT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {acc_l2:.4f} ({acc_l2*100:.2f}%)\n")
            f.write(f"F1 Score: {f1_l2:.4f}\n\n")
            f.write(classification_report(y_true_l2, y_pred_l2, zero_division=0))
            f.write("\n\n")
            
            f.write("LEVEL 3: QUERYTYPE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {acc_l3:.4f} ({acc_l3*100:.2f}%)\n")
            f.write(f"F1 Score: {f1_l3:.4f}\n\n")
            f.write(classification_report(y_true_l3, y_pred_l3, zero_division=0))
            f.write("\n\n")
            
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(summary_df.to_string(index=False))
        
        print(f"\n💾 Saved metrics → metrics_3level.txt")
        
        # ===== PLOT CONFUSION MATRICES =====
        print("\n📊 Generating confusion matrix plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Confusion Matrices - All 3 Levels', fontsize=16, fontweight='bold')
        
        # Level 1
        sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Level 1: MasterDepartment\nAccuracy: {acc_l1:.2%}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Level 2
        sns.heatmap(cm_l2, annot=False, cmap='Greens', ax=axes[1])
        axes[1].set_title(f'Level 2: Department\nAccuracy: {acc_l2:.2%}')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        # Level 3
        sns.heatmap(cm_l3, annot=False, cmap='Oranges', ax=axes[2])
        axes[2].set_title(f'Level 3: QueryType\nAccuracy: {acc_l3:.2%}')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_3level.png', dpi=150, bbox_inches='tight')
        print("💾 Saved plot → confusion_matrices_3level.png")
        plt.show()
        
        # ===== ACCURACY BAR CHART =====
        fig, ax = plt.subplots(figsize=(10, 6))
        
        levels = ['Level 1\nMasterDept', 'Level 2\nDepartment', 'Level 3\nQueryType']
        accuracies = [acc_l1 * 100, acc_l2 * 100, acc_l3 * 100]
        colors = ['#1565c0', '#2e7d32', '#e65100']
        
        bars = ax.bar(levels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('3-Level Hierarchical Model - Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison_3level.png', dpi=150, bbox_inches='tight')
        print("💾 Saved plot → accuracy_comparison_3level.png")
        plt.show()
        
    else:
        print("\n⚠️  No ground truth columns found - skipping evaluation")
        print(f"   Looking for: {TRUE_MASTERDEPT_COL}, {TRUE_DEPT_COL}, {TRUE_QUERYTYPE_COL}")
    
    print("\n" + "=" * 80)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
