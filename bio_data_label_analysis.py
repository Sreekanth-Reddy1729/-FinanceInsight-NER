#his script reads BIO-formatted training and testing files, extracts 
# unique entity labels, and generates label-to-ID 
# mappings required for consistent NER model training.
from collections import defaultdict

TRAIN_PATH = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\trainf.txt"
TEST_PATH  = r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\testf.txt"


def read_bio_file(path):
    sentences = []
    labels = []

    words = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
            else:
                token, tag = line.split()
                words.append(token)
                tags.append(tag)

        # last sentence
        if words:
            sentences.append(words)
            labels.append(tags)

    return sentences, labels


# Read files
train_sentences, train_labels = read_bio_file(TRAIN_PATH)
test_sentences, test_labels = read_bio_file(TEST_PATH)

# Collect unique labels
label_set = set()
for seq in train_labels + test_labels:
    for tag in seq:
        label_set.add(tag)

label_list = sorted(label_set)

label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Print summary
print("=== DATA SUMMARY ===")
print(f"Train sentences: {len(train_sentences)}")
print(f"Test sentences : {len(test_sentences)}")
print()

print("=== LABELS ===")
for label, idx in label2id.items():
    print(f"{label:12s} -> {idx}")
