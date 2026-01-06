# This script converts auto-labeled data from Label Studio JSON format into token-level 
# BIO tagging format using a BERT tokenizer.

import json,re
from transformers import AutoTokenizer

INPUT_FILE=r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\labelstudio_auto_labeled_full.json"
OUTPUT_FILE=r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\bio_output.txt"

tok=AutoTokenizer.from_pretrained("bert-base-uncased")

def norm(t):
    t=re.sub(r"([$€₹])\s+(\d)",r"\1\2",t)
    return re.sub(r"\s+"," ",t).strip()

with open(INPUT_FILE,encoding="utf-8") as f:
    data=json.load(f)

with open(OUTPUT_FILE,"w",encoding="utf-8") as o:
    for it in data:
        text=norm(it["data"]["text"])
        ents=[(r["value"]["start"],r["value"]["end"],r["value"]["labels"][0])
              for r in it["predictions"][0]["result"]]

        enc=tok(text,return_offsets_mapping=True,add_special_tokens=False)
        tokens=tok.convert_ids_to_tokens(enc["input_ids"])
        offs=enc["offset_mapping"]
        tags=["O"]*len(tokens)

        for s,e,l in ents:
            started=False
            for i,(ts,te) in enumerate(offs):
                if te<=s or ts>=e: continue
                tags[i]=("B-" if not started else "I-")+l
                started=True

        for i,t in enumerate(tokens):
            if t in ["$","€","₹"] and i+1<len(tags) and tags[i+1].endswith("MONEY"):
                tags[i]="B-MONEY"

        for t,tag in zip(tokens,tags):
            o.write(f"{t} {tag}\n")
        o.write("\n")

print("✅ BIO created")
