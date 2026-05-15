import re

# ── CONFIG ──────────────────────────────────────────────────────────────
INPUT_FILE  = "categories.txt"   # one category per line
OUTPUT_FILE = "choices.txt"      # output JSX lines
# ────────────────────────────────────────────────────────────────────────

def convert_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    # Replace every " > " with " &gt; "
    converted = line.replace(" > ", " &gt; ")
    return f"<Choice value='{converted}' />"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

results = []
skipped = 0

for raw in lines:
    out = convert_line(raw)
    if out:
        results.append(out)
    else:
        skipped += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"Done! {len(results)} categories converted → {OUTPUT_FILE}")
if skipped:
    print(f"({skipped} blank lines skipped)")
