#This script tokenizes BIO-formatted financial text using FinBERT 
# tokenizer and aligns word-level labels with 
# subword tokens to prepare data for model training

from transformers import AutoTokenizer
from collections import defaultdict

# ===== PATHS =====
TRAIN_PATH = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\trainf2.txt"
TEST_PATH  = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\testf2.txt"

MODEL_NAME = "ProsusAI/finbert"
#C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\trainf2.txt

# ===== LABEL MAP (from STEP 4 output) =====
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


# ===== BIO READER =====
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


# ===== TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ===== ALIGN FUNCTION =====
def tokenize_and_align_labels(sentences, labels):
    tokenized = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    aligned_labels = []

    for i, sentence_labels in enumerate(labels):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[sentence_labels[word_id]])
            else:
                label_ids.append(-100)

            prev_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized.pop("offset_mapping")
    tokenized["labels"] = aligned_labels
    return tokenized


# ===== LOAD DATA =====
train_sentences, train_labels = read_bio(TRAIN_PATH)
test_sentences, test_labels = read_bio(TEST_PATH)

train_encodings = tokenize_and_align_labels(train_sentences, train_labels)
test_encodings = tokenize_and_align_labels(test_sentences, test_labels)

# ===== SANITY CHECK =====
print("=== TOKENIZATION CHECK (First sentence) ===")
print("Tokens :", tokenizer.convert_ids_to_tokens(train_encodings["input_ids"][0]))
print("Labels :", train_encodings["labels"][0])
print("Label names:")
for l in train_encodings["labels"][0]:
    print(id2label[l] if l != -100 else "-100", end=" ")
print()
