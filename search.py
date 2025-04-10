import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load FAISS index ===
index = faiss.read_index("faiss_legal.index")

# === Load metadata ===
with open("faiss_legal_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_faiss(query, top_k=5):
    embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "file": metadata[idx]["file"],
            "text": metadata[idx]["text"][:700] + "..."
        })
    return results

# === Main loop: user input ===
if __name__ == "__main__":
    print("\n⚖️ Legal Case Search - Type your case query below (or type 'exit' to quit):\n")
    while True:
        user_query = input("📝 Enter your legal case description: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break

        print("\n🔎 Searching similar cases...\n")
        results = search_faiss(user_query, top_k=5)

        for res in results:
            print(f"\n🔹 Rank {res['rank']} | Score: {res['score']:.2f}")
            print(f"📄 File: {res['file']}")
            print(f"📜 Text:\n{res['text']}")
            print("-" * 80)

