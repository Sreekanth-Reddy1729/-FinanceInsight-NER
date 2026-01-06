#This script integrates sentence-level financial NER results with 
# automatically detected financial tables to generate a 
# unified, structured JSON output from complete financial documents.

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
import json

# =========================================================
# ================== CONFIG PATHS =========================
# =========================================================

TEXT_DOC_PATH = r"C:\Users\Admin\Desktop\project\clean\prepossedfiles\financial_doc.txt"
TABLE_DOC_PATH = r"C:\Users\Admin\Desktop\project\clean\prepossedfiles\financial_tables.txt"
MODEL_DIR = "./finbert_ner_model/checkpoint-681"

# =========================================================
# ================== LOAD FINBERT =========================
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()
id2label = model.config.id2label

# =========================================================
# ================== FINBERT NER ==========================
# =========================================================

def predict_ner(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.argmax(dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    merged = []
    word = ""
    label = None

    for tok, pred in zip(tokens, preds):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if tok.startswith("##"):
            word += tok[2:]
        else:
            if word and label != "O":
                merged.append((word, label))
            word = tok
            label = id2label[pred]

    if word and label != "O":
        merged.append((word, label))

    return merged


def group_entities(bio_entities):
    grouped = defaultdict(list)
    current_words = []
    current_label = None

    for word, tag in bio_entities:
        if tag.startswith("B-"):
            if current_words:
                grouped[current_label].append(" ".join(current_words))
            current_label = tag[2:]
            current_words = [word]

        elif tag.startswith("I-") and current_label == tag[2:]:
            current_words.append(word)
        else:
            if current_words:
                grouped[current_label].append(" ".join(current_words))
            current_words = []
            current_label = None

    if current_words:
        grouped[current_label].append(" ".join(current_words))

    return dict(grouped)

# =========================================================
# ============== DOCUMENT SEGMENTATION ====================
# =========================================================

SECTIONS = [
    "Risk Factors",
    "Profit & Loss",
    "Market & Growth",
    "Investment & Expansion",
    "Costs & Expenses",
    "Financial Performance"
]

def parse_text_document():
    sections = defaultdict(list)
    current_section = "General"

    with open(TEXT_DOC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line in SECTIONS:
                current_section = line
                continue

            if line.endswith("."):
                entities = group_entities(predict_ner(line))
                if entities:
                    sections[current_section].append({
                        "sentence": line,
                        "entities": entities
                    })

    return sections

# =========================================================
# ================= TABLE DETECTION =======================
# =========================================================

KEYWORDS = [
    "assets", "liabilities", "revenue", "income", "cash",
    "debt", "margin", "expenses", "capital", "equity"
]

def has_many_numbers(line, threshold=4):
    return sum(c.isdigit() for c in line) >= threshold

def looks_like_financial_row(line):
    line = line.lower()
    return any(k in line for k in KEYWORDS)

def detect_table_blocks(lines):
    tables = []
    current_title = None
    current_rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not any(c.isdigit() for c in line) and len(line.split()) <= 4:
            if current_title and len(current_rows) >= 2:
                tables.append((current_title, current_rows))
            current_title = line
            current_rows = []
            continue

        if has_many_numbers(line) and looks_like_financial_row(line):
            current_rows.append(line)

    if current_title and len(current_rows) >= 2:
        tables.append((current_title, current_rows))

    return tables

def parse_table(title, lines):
    rows = []
    for line in lines:
        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 2:
            rows.append({
                "item": parts[0].strip(),
                "value": parts[1].strip()
            })

    return {
        "table_type": title,
        "rows": rows
    }

def parse_all_tables():
    with open(TABLE_DOC_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    detected = detect_table_blocks(lines)
    parsed_tables = []

    table_id = 1
    for title, rows in detected:
        table = parse_table(title, rows)
        table["table_id"] = table_id
        parsed_tables.append(table)
        table_id += 1

    return parsed_tables

# =========================================================
# ================= FINAL MERGE ===========================
# =========================================================

def build_final_output():
    return {
        "document": {
            "text_sections": parse_text_document(),
            "tables": parse_all_tables()
        }
    }

# =========================================================
# ======================== RUN ============================
# =========================================================

if __name__ == "__main__":
    final_output = build_final_output()

    print("\n=== FINAL DOCUMENT-LEVEL STRUCTURED OUTPUT ===\n")
    print(json.dumps(final_output, indent=2))

    with open("final_document_output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\n✅ TASK 7–9 COMPLETED")
