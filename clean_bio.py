#This script cleans and normalizes BIO-formatted NER data by merging subword tokens (##) 
# and fixing sentence boundaries.
#It produces a clean BIO file ready for stable model training.
IN=r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\bio_output.txt"
OUT=r"C:\Users\Admin\OneDrive\Desktop\project\clean\prepossedfiles\bio_cleaned.txt"

sents=[];cur=[]
for l in open(IN,encoding="utf-8"):
    l=l.strip()
    if not l:
        if cur: sents.append(cur); cur=[]
        continue
    t,tag=l.split()
    if t.startswith("##") and cur:
        p,pt=cur.pop()
        cur.append((p+t[2:],pt))
    else:
        cur.append((t,tag))
if cur: sents.append(cur)

with open(OUT,"w",encoding="utf-8") as o:
    for s in sents:
        for t,tag in s: o.write(f"{t} {tag}\n")
        o.write("\n")

print("✅ BIO cleaned")
