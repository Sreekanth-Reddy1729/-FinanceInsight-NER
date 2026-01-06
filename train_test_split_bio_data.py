#This script randomly splits cleaned BIO-formatted 
# financial NER data into training and testing datasets 
# required for FinBERT model fine-tuning.

import random

IN=r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\bio_cleaned.txt"

s=open(IN,encoding="utf-8").read().strip().split("\n\n")
random.shuffle(s)
k=int(0.9*len(s))

open("train.txt","w",encoding="utf-8").write("\n\n".join(s[:k]))
open("test.txt","w",encoding="utf-8").write("\n\n".join(s[k:]))

print("✅ train.txt & test.txt created")
