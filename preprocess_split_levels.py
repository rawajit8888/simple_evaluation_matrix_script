import pandas as pd

"""
PREPROCESSING SCRIPT
Splits single classification_true column into 3 separate truth columns
"""

# =========================
# CONFIG
# =========================
INPUT_CSV  = "emails.csv"                      # Your original CSV
OUTPUT_CSV = "emails_with_3levels.csv"         # Output with split columns

CLASSIFICATION_COL = "classification_true"     # Your original column with full path

# New columns that will be created
MASTERDEPT_COL = "masterdepartment_true"       # Level 1
DEPT_COL       = "department_true"              # Level 2
QUERYTYPE_COL  = "querytype_true"               # Level 3


# =========================
# LOAD CSV
# =========================
print("=" * 80)
print("📂 PREPROCESSING: Split Classification Into 3 Levels")
print("=" * 80)

print(f"\n📁 Loading CSV: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"✓ Loaded {len(df)} rows")

if CLASSIFICATION_COL not in df.columns:
    print(f"\n❌ ERROR: Column '{CLASSIFICATION_COL}' not found!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)


# =========================
# SPLIT FUNCTION
# =========================
def split_classification(full_path):
    """
    Split "Internet Banking > Account Access > Unblock" into 3 levels:
    
    Level 1: "Internet Banking"
    Level 2: "Internet Banking > Account Access"  
    Level 3: "Internet Banking > Account Access > Unblock"
    
    Returns: (level1, level2, level3)
    """
    if pd.isna(full_path) or not full_path.strip():
        return "", "", ""
    
    # Split by " > "
    parts = [p.strip() for p in str(full_path).split(">")]
    
    if len(parts) >= 3:
        # Full 3-level path
        level1 = parts[0]
        level2 = f"{parts[0]} > {parts[1]}"
        level3 = f"{parts[0]} > {parts[1]} > {parts[2]}"
        return level1, level2, level3
    
    elif len(parts) == 2:
        # Only 2 levels (Department level)
        level1 = parts[0]
        level2 = f"{parts[0]} > {parts[1]}"
        level3 = ""  # No query type
        return level1, level2, level3
    
    elif len(parts) == 1:
        # Only 1 level (MasterDepartment only)
        level1 = parts[0]
        level2 = ""
        level3 = ""
        return level1, level2, level3
    
    else:
        return "", "", ""


# =========================
# APPLY SPLIT
# =========================
print(f"\n🔄 Splitting '{CLASSIFICATION_COL}' into 3 levels...")

# Create the 3 new columns
df[[MASTERDEPT_COL, DEPT_COL, QUERYTYPE_COL]] = df[CLASSIFICATION_COL].apply(
    lambda x: pd.Series(split_classification(x))
)

print("✓ Split complete!")


# =========================
# SHOW SAMPLE
# =========================
print("\n📊 Sample rows (showing first 5):")
print("=" * 80)

sample_cols = [CLASSIFICATION_COL, MASTERDEPT_COL, DEPT_COL, QUERYTYPE_COL]
available_cols = [col for col in sample_cols if col in df.columns]

print(df[available_cols].head().to_string(index=False))


# =========================
# STATISTICS
# =========================
print("\n" + "=" * 80)
print("📈 STATISTICS")
print("=" * 80)

total = len(df)
level3_count = df[QUERYTYPE_COL].notna().sum() if QUERYTYPE_COL in df.columns else 0
level2_count = df[DEPT_COL].notna().sum() if DEPT_COL in df.columns else 0
level1_count = df[MASTERDEPT_COL].notna().sum() if MASTERDEPT_COL in df.columns else 0

# Count non-empty strings
level3_count = (df[QUERYTYPE_COL] != "").sum()
level2_count = (df[DEPT_COL] != "").sum()
level1_count = (df[MASTERDEPT_COL] != "").sum()

print(f"Total rows               : {total}")
print(f"Level 1 (MasterDept)     : {level1_count} ({level1_count/total*100:.1f}%)")
print(f"Level 2 (Department)     : {level2_count} ({level2_count/total*100:.1f}%)")
print(f"Level 3 (QueryType)      : {level3_count} ({level3_count/total*100:.1f}%)")

print("\nUnique categories per level:")
print(f"  Level 1: {df[MASTERDEPT_COL].nunique()} unique MasterDepartments")
print(f"  Level 2: {df[DEPT_COL].nunique()} unique Departments")
print(f"  Level 3: {df[QUERYTYPE_COL].nunique()} unique QueryTypes")


# =========================
# VALIDATION CHECK
# =========================
print("\n" + "=" * 80)
print("🔍 VALIDATION CHECK")
print("=" * 80)

# Check for rows where split might have failed
issues = df[
    (df[CLASSIFICATION_COL] != "") & 
    (df[MASTERDEPT_COL] == "")
]

if len(issues) > 0:
    print(f"⚠️  Found {len(issues)} rows where split failed:")
    print(issues[[CLASSIFICATION_COL]].head())
else:
    print("✓ All non-empty rows split successfully!")


# =========================
# SAVE OUTPUT
# =========================
print(f"\n💾 Saving to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print("✓ Saved!")

print("\n" + "=" * 80)
print("✅ PREPROCESSING COMPLETE")
print("=" * 80)
print(f"\nNext step: Use '{OUTPUT_CSV}' with the inference script")
print("The inference script will now find these columns:")
print(f"  - {MASTERDEPT_COL}")
print(f"  - {DEPT_COL}")
print(f"  - {QUERYTYPE_COL}")
