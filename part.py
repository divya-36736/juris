import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === CONFIG ===
FAISS_INDEX_PATH = "faiss_legal.index"
METADATA_PATH = "faiss_legal_metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# === Load FAISS index and metadata ===
print("🔁 Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Load embedding model ===
embed_model = SentenceTransformer(EMBED_MODEL)

# === Load LLM (Qwen 1.5B Instruct) ===
print("🧠 Loading Qwen2.5-1.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, trust_remote_code=True, device_map="auto")
qwen_chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Semantic search ===
def search_faiss(query, top_k=5):
    embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(embedding, top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "file": metadata[idx]["file"],
            "text": metadata[idx]["text"][:1000]
        })
    return results

# === CoT + Reflexion prompt ===
def build_prompt(user_case, similar_cases):
    cases_text = "\n\n".join(
        f"CASE {i+1}:\n{case['text']}" for i, case in enumerate(similar_cases)
    )
    return f"""
You are a legal AI trained on Indian case law. Use Chain of Thought reasoning and reflect on your answer.

USER CASE:
{user_case}

SIMILAR PRECEDENT CASES:
{cases_text}

TASK:
1. Analyze the user case facts.
2. Compare with the precedents step-by-step.
3. Use Chain of Thought reasoning to assess strength.
4. Predict outcomes based on legal patterns.
5. Reflect on your reasoning and revise if necessary.
6. Provide final structured legal advice.

Respond with bullet points or numbered legal arguments.
""".strip()

# === Reflexion step ===
def run_reflexion(prompt):
    first_response = qwen_chat(prompt, max_new_tokens=700, do_sample=True, temperature=0.3)[0]["generated_text"]

    reflexion_prompt = f"""
Your previous response was:

{first_response}

Reflect on this answer. Are there missing steps, incorrect assumptions, or stronger arguments to add?
Revise and improve the analysis with better legal logic and structure.
""".strip()

    revised_response = qwen_chat(reflexion_prompt, max_new_tokens=700, do_sample=True, temperature=0.3)[0]["generated_text"]
    return first_response, revised_response

# === Main interactive loop ===
if __name__ == "__main__":
    print("\n⚖️ Legal Advisor – CoT + Reflexion Enabled\n")

    while True:
        user_input = input("📥 Describe your legal case (or type 'exit'): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting Legal Advisor.")
            break

        print("🔍 Searching similar cases...")
        top_cases = search_faiss(user_input, top_k=5)

        print("🧠 Generating analysis with CoT reasoning...")
        prompt = build_prompt(user_input, top_cases)
        original, improved = run_reflexion(prompt)

        print("\n🧩 ORIGINAL RESPONSE (Chain of Thought):\n")
        print(original)
        print("\n🔁 REFLECTED & IMPROVED LEGAL ADVICE:\n")
        print(improved)
        print("\n" + "=" * 100 + "\n")

