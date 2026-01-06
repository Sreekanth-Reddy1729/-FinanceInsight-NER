#This script evaluates the trained FinBERT NER model on unseen test data and generates 
# per-entity precision, recall, and F1-score reports.

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer
)
from seqeval.metrics import classification_report


# ================= PATHS =================
TEST_PATH = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\testf2.txt"
#MODEL_DIR = "./finbert_ner_model" 
     # trained model folder
MODEL_DIR = "./finbert_ner_model/checkpoint-675"

BASE_MODEL = "ProsusAI/finbert"


# ================= LABEL MAP (MUST MATCH TRAINING) =================
label2id = {
    "B-DATE": 0,
    "B-EVENT": 1,
    "B-LOC": 2,
    "B-METRIC": 3,
    "B-MONEY": 4,
    "B-ORG": 5,
    "B-PERCENT": 6,
    "B-PERSON": 7,
    "B-PRODUCT": 8,
    "I-DATE": 9,
    "I-LOC": 10,
    "I-MONEY": 11,
    "I-ORG": 12,
    "I-PERCENT": 13,
    "I-PERSON": 14,
    "I-PRODUCT": 15,
    "O": 16
}
id2label = {v: k for k, v in label2id.items()}


# ================= BIO FILE READER =================
def read_bio(path):
    sentences, labels = [], []
    words, tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                token, tag = line.split()
                words.append(token)
                tags.append(tag)

        if words:
            sentences.append(words)
            labels.append(tags)

    return sentences, labels


# ================= LOAD TEST DATA =================
test_sentences, test_labels = read_bio(TEST_PATH)

test_dataset = Dataset.from_dict({
    "tokens": test_sentences,
    "ner_tags": test_labels
})


# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[label_seq[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


test_dataset = test_dataset.map(tokenize_and_align, batched=True)
test_dataset.set_format("torch")


# ================= LOAD TRAINED MODEL =================
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)


# ================= TRAINER (NO TRAINING) =================
trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)


# ================= EVALUATION =================
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

true_preds = [
    [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
    for pred, lab in zip(predictions, labels)
]

true_labels = [
    [id2label[l] for l in lab if l != -100]
    for lab in labels
]


print("\n=== PER-ENTITY CLASSIFICATION REPORT ===\n")
print(classification_report(true_labels, true_preds))
