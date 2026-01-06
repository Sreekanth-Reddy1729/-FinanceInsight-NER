#This script fine-tunes the FinBERT model on BIO-labeled financial text using
#  Hugging Face Trainer to learn financial named entity recognition.


import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# ===== PATHS =====
TRAIN_PATH = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\trainf2.txt"
TEST_PATH  = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\testf2.txt"
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "./finbert_ner_model"


# ===== LABEL MAP =====
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


# ===== LOAD DATA =====
train_sentences, train_labels = read_bio(TRAIN_PATH)
test_sentences, test_labels = read_bio(TEST_PATH)

train_dataset = Dataset.from_dict({
    "tokens": train_sentences,
    "ner_tags": train_labels
})

test_dataset = Dataset.from_dict({
    "tokens": test_sentences,
    "ner_tags": test_labels
})


# ===== TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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


train_dataset = train_dataset.map(tokenize_and_align, batched=True)
test_dataset = test_dataset.map(tokenize_and_align, batched=True)

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# ===== MODEL =====
# model = AutoModelForTokenClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id=label2id
# )
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)



# ===== METRICS =====
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for l in lab if l != -100]
        for lab in labels
    ]

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


# ===== TRAINING ARGS =====
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
#     report_to="none"
# )
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)



# ===== TRAINER =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# ===== TRAIN =====
trainer.train()

# ===== FINAL EVALUATION =====
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

print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(true_labels, true_preds))

# import numpy as np
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer
# )
# from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# # ================= PATHS =================
# TRAIN_PATH = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\trainf2.txt"
# TEST_PATH  = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\testf2.txt"
# MODEL_NAME = "ProsusAI/finbert"
# OUTPUT_DIR = "./finbert_ner_model"


# # ================= LABEL MAP =================
# label2id = {
#     "B-DATE": 0,
#     "B-EVENT": 1,
#     "B-LOC": 2,
#     "B-METRIC": 3,
#     "B-MONEY": 4,
#     "B-ORG": 5,
#     "B-PERCENT": 6,
#     "B-PERSON": 7,
#     "B-PRODUCT": 8,
#     "I-DATE": 9,
#     "I-LOC": 10,
#     "I-MONEY": 11,
#     "I-ORG": 12,
#     "I-PERCENT": 13,
#     "I-PERSON": 14,
#     "I-PRODUCT": 15,
#     "O": 16
# }
# id2label = {v: k for k, v in label2id.items()}


# # ================= BIO FILE READER =================
# def read_bio(path):
#     sentences, labels = [], []
#     words, tags = [], []

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line == "":
#                 if words:
#                     sentences.append(words)
#                     labels.append(tags)
#                     words, tags = [], []
#             else:
#                 token, tag = line.split()
#                 words.append(token)
#                 tags.append(tag)

#         if words:
#             sentences.append(words)
#             labels.append(tags)

#     return sentences, labels


# # ================= LOAD DATA =================
# train_sentences, train_labels = read_bio(TRAIN_PATH)
# test_sentences, test_labels = read_bio(TEST_PATH)

# train_dataset = Dataset.from_dict({
#     "tokens": train_sentences,
#     "ner_tags": train_labels
# })

# test_dataset = Dataset.from_dict({
#     "tokens": test_sentences,
#     "ner_tags": test_labels
# })


# # ================= TOKENIZER =================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# def tokenize_and_align(examples):
#     tokenized = tokenizer(
#         examples["tokens"],
#         is_split_into_words=True,
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )

#     labels = []
#     for i, label_seq in enumerate(examples["ner_tags"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         prev_word_id = None
#         label_ids = []

#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != prev_word_id:
#                 label_ids.append(label2id[label_seq[word_id]])
#             else:
#                 label_ids.append(-100)

#             prev_word_id = word_id

#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized


# train_dataset = train_dataset.map(tokenize_and_align, batched=True)
# test_dataset = test_dataset.map(tokenize_and_align, batched=True)

# train_dataset.set_format("torch")
# test_dataset.set_format("torch")


# # ================= MODEL =================
# model = AutoModelForTokenClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True   # 🔥 REQUIRED FOR FINBERT
# )


# # ================= METRICS =================
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_preds = [
#         [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
#         for pred, lab in zip(predictions, labels)
#     ]
#     true_labels = [
#         [id2label[l] for l in lab if l != -100]
#         for lab in labels
#     ]

#     return {
#         "precision": precision_score(true_labels, true_preds),
#         "recall": recall_score(true_labels, true_preds),
#         "f1": f1_score(true_labels, true_preds),
#     }


# # ================= TRAINING ARGS =================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     eval_strategy="epoch",          # ✅ FIXED FOR NEW TRANSFORMERS
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
#     report_to="none"
# )


# # ================= TRAINER =================
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )


# # ================= TRAIN =================
# trainer.train()


# # ================= FINAL EVALUATION =================
# predictions, labels, _ = trainer.predict(test_dataset)
# predictions = np.argmax(predictions, axis=2)

# true_preds = [
#     [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
#     for pred, lab in zip(predictions, labels)
# ]
# true_labels = [
#     [id2label[l] for l in lab if l != -100]
#     for lab in labels
# ]

# print("\n=== FINAL CLASSIFICATION REPORT ===")
# print(classification_report(true_labels, true_preds))


#new...

# import numpy as np
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer
# )
# from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# # ================= PATHS =================
# TRAIN_PATH = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\trainf.txt"
# TEST_PATH  = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\testf.txt"
# MODEL_NAME = "ProsusAI/finbert"
# OUTPUT_DIR = "./finbert_ner_model"


# # ================= LABEL MAP =================
# label2id = {
#     "B-DATE": 0,
#     "B-EVENT": 1,
#     "B-LOC": 2,
#     "B-METRIC": 3,
#     "B-MONEY": 4,
#     "B-ORG": 5,
#     "B-PERCENT": 6,
#     "B-PERSON": 7,
#     "B-PRODUCT": 8,
#     "I-DATE": 9,
#     "I-LOC": 10,
#     "I-MONEY": 11,
#     "I-ORG": 12,
#     "I-PERCENT": 13,
#     "I-PERSON": 14,
#     "I-PRODUCT": 15,
#     "O": 16
# }
# id2label = {v: k for k, v in label2id.items()}


# # ================= BIO FILE READER =================
# def read_bio(path):
#     sentences, labels = [], []
#     words, tags = [], []

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line == "":
#                 if words:
#                     sentences.append(words)
#                     labels.append(tags)
#                     words, tags = [], []
#             else:
#                 token, tag = line.split()
#                 words.append(token)
#                 tags.append(tag)

#         if words:
#             sentences.append(words)
#             labels.append(tags)

#     return sentences, labels


# # ================= LOAD DATA =================
# train_sentences, train_labels = read_bio(TRAIN_PATH)
# test_sentences, test_labels = read_bio(TEST_PATH)

# train_dataset = Dataset.from_dict({
#     "tokens": train_sentences,
#     "ner_tags": train_labels
# })
# test_dataset = Dataset.from_dict({
#     "tokens": test_sentences,
#     "ner_tags": test_labels
# })


# # ================= TOKENIZER =================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# def tokenize_and_align(examples):
#     tokenized = tokenizer(
#         examples["tokens"],
#         is_split_into_words=True,
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )

#     labels = []
#     for i, label_seq in enumerate(examples["ner_tags"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         prev_word_id = None
#         label_ids = []

#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != prev_word_id:
#                 label_ids.append(label2id[label_seq[word_id]])
#             else:
#                 label_ids.append(-100)
#             prev_word_id = word_id

#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized


# train_dataset = train_dataset.map(tokenize_and_align, batched=True)
# test_dataset = test_dataset.map(tokenize_and_align, batched=True)

# train_dataset.set_format("torch")
# test_dataset.set_format("torch")


# # ================= MODEL =================
# model = AutoModelForTokenClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True
# )


# # ================= METRICS =================
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_preds = [
#         [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
#         for pred, lab in zip(predictions, labels)
#     ]
#     true_labels = [
#         [id2label[l] for l in lab if l != -100]
#         for lab in labels
#     ]

#     return {
#         "precision": precision_score(true_labels, true_preds),
#         "recall": recall_score(true_labels, true_preds),
#         "f1": f1_score(true_labels, true_preds),
#     }


# # ================= TRAINING ARGS =================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,

#     # 🔥 KEY FIX: TOTAL epochs must be > completed epochs
#     num_train_epochs=2,

#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
#     report_to="none"
# )
# # training_args = TrainingArguments(
# #     output_dir=OUTPUT_DIR,
# #     eval_strategy="epoch",        # ✅ FIXED
# #     save_strategy="epoch",
# #     learning_rate=2e-5,
# #     per_device_train_batch_size=8,
# #     per_device_eval_batch_size=8,
# #     num_train_epochs=2,
# #     weight_decay=0.01,
# #     logging_dir="./logs",
# #     logging_steps=50,
# #     load_best_model_at_end=True,
# #     metric_for_best_model="f1",
# #     greater_is_better=True,
# #     report_to="none"
# # )



# # ================= TRAINER =================
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )


# # ================= TRAIN (RESUME CORRECTLY) =================
# trainer.train(resume_from_checkpoint="./finbert_ner_model/checkpoint-675")


# # ================= FINAL EVALUATION =================
# predictions, labels, _ = trainer.predict(test_dataset)
# predictions = np.argmax(predictions, axis=2)

# true_preds = [
#     [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
#     for pred, lab in zip(predictions, labels)
# ]
# true_labels = [
#     [id2label[l] for l in lab if l != -100]
#     for lab in labels
# ]

# print("\n=== FINAL CLASSIFICATION REPORT ===")
# print(classification_report(true_labels, true_preds))
