# %%
# Optional: Authenticate with Hugging Face if needed
# from huggingface_hub import login
# login(token="YOUR_HUGGINGFACE_TOKEN_HERE")

# %%
import pandas as pd
import math
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# params
tau = 0.5
lam = 0.7

def pairwise_adjusted_score(sim, tau=tau, lam=lam):
    penalty = lam * max(0, tau - sim)
    return float(sim - penalty)

# Load class names from CSV file
df = pd.read_csv("label_map.csv")  # Column should contain class names
classes = df["classname"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# model = SentenceTransformer("google/embeddinggemma-300m")
# model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# emb_all = model.encode(classes)
emb_all = model.encode(
    classes,
    normalize_embeddings=True,
    batch_size=32
)

N = len(classes)
rankings = {}

for i in range(N):
    gt = classes[i]
    gt_emb = emb_all[i]
    sims = cosine_similarity([gt_emb], emb_all)[0]   # sims[j] is sim(gt, classes[j])

    entries = []
    for j in range(N):
        if j == i:
            score = 1.0
        else:
            score = pairwise_adjusted_score(sims[j])
        entries.append((classes[j], score))

    # sort except ensure GT is rank 1
    entries_sorted = sorted([e for e in entries if e[0] != gt], key=lambda x: x[1], reverse=True)
    entries_sorted.insert(0, (gt, 1.0))

    # build JSON-friendly list with ranks
    rankings[gt] = [
        {"rank": idx + 1, "class": cname, "score": float(score)}
        for idx, (cname, score) in enumerate(entries_sorted)
    ]

# %% save
with open("rankings.json", "w") as f:
    json.dump(rankings, f, indent=2)

# %%



