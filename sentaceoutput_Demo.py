# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from collections import defaultdict

# # ================== MODEL LOAD ==================
# MODEL_DIR = "./finbert_ner_model/checkpoint-681"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
# model.eval()

# id2label = model.config.id2label

# # ================== ENTITY EXTRACTION ==================
# def extract_entities(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     preds = outputs.logits.argmax(dim=2)[0].tolist()
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

#     entities = []
#     current_words = []
#     current_label = None

#     for tok, pred in zip(tokens, preds):
#         label = id2label[pred]

#         if tok in ["[CLS]", "[SEP]", "[PAD]"]:
#             continue

#         if tok.startswith("##"):
#             if current_words:
#                 current_words[-1] += tok[2:]
#             continue

#         if label.startswith("B-"):
#             if current_words:
#                 entities.append((" ".join(current_words), current_label))
#             current_words = [tok]
#             current_label = label[2:]

#         elif label.startswith("I-") and current_label == label[2:]:
#             current_words.append(tok)

#         else:
#             if current_words:
#                 entities.append((" ".join(current_words), current_label))
#             current_words = []
#             current_label = None

#     if current_words:
#         entities.append((" ".join(current_words), current_label))

#     grouped = defaultdict(list)
#     for ent, lbl in entities:
#         grouped[lbl].append(ent)

#     return grouped

# # ================== PRETTY PRINT ==================
# def pretty_print(section_name, data):
#     label_map = {
#         "ORG": "Company",
#         "MONEY": "Amount",
#         "LOC": "Location",
#         "DATE": "Date",
#         "PERCENT": "Percentage",
#         "METRIC": "Metric",
#         "EVENT": "Event"
#     }

#     if not data:
#         return

#     print(f"\n>> {section_name}")
#     print("-" * 45)

#     for label, values in data.items():
#         readable = label_map.get(label, label)
#         clean_values = [" ".join(v.split()).title() for v in values]
#         print(f"{readable:15s}: {', '.join(clean_values)}")

#     print("-" * 45)

# # ================== DOCUMENT PROCESSING ==================
# with open("financial_doc.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()

# financial_section = []
# risk_section = []
# other_section = []

# for line in lines:
#     lower = line.lower()

#     if any(word in lower for word in ["revenue", "profit", "usd", "eur", "eps"]):
#         financial_section.append(line)

#     elif "risk" in lower or "uncertain" in lower:
#         risk_section.append(line)

#     else:
#         other_section.append(line)

# print("\n=== Milestone 4: Document-Level Financial Extraction ===")

# # -------- FINANCIAL SECTION --------
# for line in financial_section:
#     result = extract_entities(line)
#     pretty_print("Financial Section Extraction", result)

# # -------- RISK SECTION --------
# print("\n>> Risk Section")
# print("-" * 45)
# for line in risk_section:
#     print(line.strip())
# print("-" * 45)

# -------- OTHER SECTION --------
# for line in other_section:
#     result = extract_entities(line)
#     pretty_print("Other Section Extraction", result)

#new

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

# ================= MODEL LOAD =================
MODEL_DIR = "./finbert_ner_model/checkpoint-681"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label


# ================= ENTITY EXTRACTION =================
def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.argmax(dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    current_words = []
    current_label = None

    for tok, pred in zip(tokens, preds):
        label = id2label[pred]

        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if tok.startswith("##"):
            if current_words:
                current_words[-1] += tok[2:]
            continue

        if label.startswith("B-"):
            if current_words:
                entities.append((" ".join(current_words), current_label))
            current_words = [tok]
            current_label = label[2:]

        elif label.startswith("I-") and current_label == label[2:]:
            current_words.append(tok)

        else:
            if current_words:
                entities.append((" ".join(current_words), current_label))
            current_words = []
            current_label = None

    if current_words:
        entities.append((" ".join(current_words), current_label))

    grouped = defaultdict(list)
    for ent, lbl in entities:
        grouped[lbl].append(ent)

    return grouped


# ================= PRETTY PRINT =================
def pretty_print(section_name, data):
    label_map = {
        "ORG": "Company",
        "MONEY": "Amount",
        "LOC": "Location",
        "DATE": "Date",
        "PERCENT": "Percentage",
        "METRIC": "Metric",
        "EVENT": "Event"
    }

    if not data:
        return

    print(f"\n>> {section_name}")
    print("-" * 45)

    for label, values in data.items():
        readable = label_map.get(label, label)
        clean_values = [" ".join(v.split()).title() for v in values]
        print(f"{readable:15s}: {', '.join(clean_values)}")

    print("-" * 45)


# ================= DOCUMENT LOAD =================
with open("financial_doc.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

financial_section = []
risk_section = []
other_section = []

for line in lines:
    lower = line.lower()

    if any(w in lower for w in ["revenue", "profit", "usd", "eur", "eps"]):
        financial_section.append(line)

    elif "risk" in lower or "uncertain" in lower:
        risk_section.append(line)

    else:
        other_section.append(line)


# ================= FINAL OUTPUT =================
print("\n=== Milestone 4: Document-Level Financial Extraction ===")

# -------- FINANCIAL SECTION (COMBINED ONCE) --------
combined_financial = defaultdict(list)

for line in financial_section:
    result = extract_entities(line)
    for label, values in result.items():
        combined_financial[label].extend(values)

pretty_print("Financial Section Extraction", combined_financial)


# -------- RISK SECTION --------
print("\n>> Risk Section")
print("-" * 45)
for line in risk_section:
    print(line.strip())
print("-" * 45)


# -------- OTHER SECTION (COMBINED ONCE) --------
combined_other = defaultdict(list)

for line in other_section:
    result = extract_entities(line)
    for label, values in result.items():
        combined_other[label].extend(values)

pretty_print("Other Section Extraction", combined_other)




