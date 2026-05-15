import csv
import unicodedata
import sys
import os

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    replacements = {
        '\u00a0': ' ',   # non-breaking space
        '\u200b': '',    # zero-width space
        '\u200c': '',    # zero-width non-joiner
        '\u200d': '',    # zero-width joiner
        '\ufeff': '',    # BOM
        '\uff1e': '>',   # fullwidth >
        '\uff1c': '<',   # fullwidth <
        '\u2019': "'",
        '\u2018': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2013': '-',
        '\u2014': '-',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = ' '.join(text.split())
    return text.strip()

def to_xml_escape(text):
    text = text.replace('&', '&amp;')
    text = text.replace('>', '&gt;')
    text = text.replace('<', '&lt;')
    text = text.replace('"', '&quot;')
    return text

def process_file(input_path):
    base, ext = os.path.splitext(input_path)
    ext = ext.lower()
    categories = []

    if ext == '.csv':
        with open(input_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    # If multiple columns, join with ' > '
                    line = ' > '.join(cell.strip() for cell in row if cell.strip())
                    if line:
                        categories.append(line)

    elif ext == '.txt':
        with open(input_path, encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line:
                    categories.append(line)

    else:
        print(f"Unsupported file type: {ext}. Please use .csv or .txt")
        sys.exit(1)

    print(f"\nFound {len(categories)} categories in '{input_path}'")

    # Clean and deduplicate
    seen = set()
    unique_cleaned = []
    for cat in categories:
        clean = clean_text(cat)
        if clean and clean not in seen:
            seen.add(clean)
            unique_cleaned.append(clean)

    print(f"After cleaning: {len(unique_cleaned)} unique categories")

    # Write output XML
    output_path = base + "_labelstudio.xml"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<View>\n')
        f.write('  <Choices name="category" toName="text" choice="single">\n')
        for cat in unique_cleaned:
            escaped = to_xml_escape(cat)
            f.write(f'    <Choice value="{escaped}" />\n')
        f.write('  </Choices>\n')
        f.write('</View>\n')

    print(f"\nDone! Output saved to: {output_path}")
    print("\nPreview (first 5):")
    for cat in unique_cleaned[:5]:
        print(f'  <Choice value="{to_xml_escape(cat)}" />')
    if len(unique_cleaned) > 5:
        print(f"  ... and {len(unique_cleaned) - 5} more")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fix_labelstudio_choices.py categories.csv")
        print("  python fix_labelstudio_choices.py categories.txt")
        sys.exit(1)

    process_file(sys.argv[1])
