#This script loads a fine-tuned FinBERT NER model and performs real-time 
# financial entity recognition on user input text.


#It outputs detected entities with their labels in an interactive CLI demo.
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

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


print("\n=== FinBERT Financial NER Demo ===")
print("Type 'exit' to quit\n")

while True:
    text = input("Input: ")
    if text.lower() == "exit":
        break

    entities = predict(text)

    print("\nEntities:")
    for ent, lbl in entities:
        print(f"{ent:12s} → {lbl}")
    print("-" * 40)
