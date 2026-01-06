#This script loads a fine-tuned FinBERT NER model and performs document-level
#  financial entity extraction by segmenting text into sections and producing 
# structured entity outputs for each sentence.
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

# ================= MODEL PATH =================
MODEL_DIR = "./finbert_ner_model/checkpoint-681"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label

# ================= FILE PATH =================
DOC_PATH = r"C:\Users\Admin\Desktop\project\clean\prepossedfiles\financial_doc.txt"

# ================= SECTION HEADERS =================
SECTION_HEADERS = [
    "Risk Factors",
    "Profit & Loss",
    "Market & Growth",
    "Investment & Expansion",
    "Costs & Expenses",
    "Financial Performance"
]

# ================= NER PREDICTION =================
def predict(text):
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

# ================= BIO GROUPING =================
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

# ================= DOCUMENT SEGMENTATION =================
sections = defaultdict(list)
current_section = "General"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        if line in SECTION_HEADERS:
            current_section = line
            continue

        # Only process proper sentences
        if line.endswith("."):
            sections[current_section].append(line)

# ================= PARSING DOCUMENT =================
final_output = {}

for section, sentences in sections.items():
    final_output[section] = []

    for sent in sentences:
        bio_entities = predict(sent)
        parsed_json = group_entities(bio_entities)

        if parsed_json:  # store only if something is extracted
            final_output[section].append({
                "sentence": sent,
                "entities": parsed_json
            })

# ================= FINAL OUTPUT =================
print("\n=== FINAL DOCUMENT-LEVEL PARSING OUTPUT ===\n")

for section, items in final_output.items():
    print(f"\n[{section}]")
    for item in items:
        print(item)

print("\n Milestone 4 (Document Segmentation & Parsing) COMPLETED")
