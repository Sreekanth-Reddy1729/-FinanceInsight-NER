# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# # ================= PATH =================
# MODEL_DIR = "./finbert_ner_model/checkpoint-681"

# # ================= LOAD =================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
# model.eval()

# id2label = model.config.id2label

# # ================= PREDICT =================
# def predict(text):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)

#     preds = outputs.logits.argmax(dim=2)[0].tolist()
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

#     results = []
#     for token, pred in zip(tokens, preds):
#         if token in ["[CLS]", "[SEP]", "[PAD]"]:
#             continue

#         label = id2label[pred]

#         # clean wordpieces
#         if token.startswith("##"):
#             token = token[2:]

#         if label != "O":
#             results.append((token, label))

#     return results

# # ================= INTERACTIVE =================
# print("\n=== FinBERT NER TOKEN-LEVEL DEMO ===")
# print("Type 'exit' to quit\n")

# while True:
#     text = input("Input: ")
#     if text.lower() == "exit":
#         break

#     output = predict(text)

#     print("\nEntities:")
#     for tok, lbl in output:
#         print(f"{tok:10s} → {lbl}")
#     print("-" * 40)


import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

MODEL_DIR = "./finbert_ner_model/checkpoint-681"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label


def predict(text):
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

    return entities


print("\n=== FinBERT Financial NER Demo ===")
print("Type 'exit' to quit\n")

while True:
    text = input("Input: ")
    if text.lower() == "exit":
        break

    entities = predict(text)

    grouped = defaultdict(list)
    for ent, lbl in entities:
        grouped[lbl].append(ent)

    print("\n Financial Information (User View):")
    for k, v in grouped.items():
        print(f"{k}: {v}")

    print("-" * 40)
