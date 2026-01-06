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

# ================= BIO GROUPING LOGIC =================
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
            current_wordsappend(word)

        else:
            if current_words:
                grouped[current_label].append(" ".join(current_words))
                current_words = []
                current_label = None

    if current_words:
        grouped[current_label].append(" ".join(current_words))

    return dict(grouped)

# ================= INTERACTIVE DEMO =================
print("\n=== FinBERT Financial NER (Grouped JSON Output) ===")
print("Type 'exit' to quit\n")

while True:
    text = input("Input: ")
    if text.lower() == "exit":
        break

    bio_entities = predict(text)
    final_output = group_entities(bio_entities)

    print("\nStructured Financial Record (JSON):")
    print(final_output)
    print("-" * 50)
