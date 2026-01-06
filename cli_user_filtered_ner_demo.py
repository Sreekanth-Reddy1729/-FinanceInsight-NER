import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

# ================= MODEL PATH =================
MODEL_DIR = "./finbert_ner_model/checkpoint-681"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label

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

# ================= USER MENU =================
def get_user_interest():
    print("\nChoose your interest:")
    print("1. Revenue")
    print("2. Profit")
    print("3. Organization")
    print("4. All")

    choice = input("Enter choice (1-4): ").strip()

    mapping = {
        "1": "revenue",
        "2": "profit",
        "3": "org",
        "4": "all"
    }

    return mapping.get(choice, None)

# ================= FILTER LOGIC (FIXED) =================
def filter_by_user(full_json, interest):
    if interest == "all":
        return full_json

    if interest == "org":
        return {"ORG": full_json.get("ORG", [])}

    if interest in ["revenue", "profit"]:
        filtered = {}

        metrics = full_json.get("METRIC", [])
        money = full_json.get("MONEY", [])

        for m, v in zip(metrics, money):
            if interest in m.lower():   # substring match
                filtered.setdefault("METRIC", []).append(m)
                filtered.setdefault("MONEY", []).append(v)

        return filtered

    return {}

# ================= INTERACTIVE DEMO =================
print("\n=== FinBERT Financial NER (FINAL WORKING SYSTEM) ===")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter sentence: ")
    if text.lower() == "exit":
        break

    interest = get_user_interest()
    if interest is None:
        print("❌ Invalid choice. Please select 1–4.")
        continue

    bio_entities = predict(text)
    full_json = group_entities(bio_entities)
    final_json = filter_by_user(full_json, interest)

    print("\nFinal JSON for User:")
    print(final_json)
    print("-" * 60)
