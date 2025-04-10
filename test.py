import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === CONFIG ===
FAISS_INDEX_PATH = "faiss_legal.index"
METADATA_PATH = "faiss_legal_metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLAMA_MODEL = "llama3-70b-8192"
GROQ_API_KEY = "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" 

# === Load FAISS index and metadata ===
print("\n🔁 Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Load embedding model ===
embed_model = SentenceTransformer(EMBED_MODEL)

# === Setup OpenAI client for Groq ===
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

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

# === Chat with Groq (LLM model) ===
def chat_call(messages):
    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# === CoT + Reflexion prompt ===
def build_prompt(user_case, similar_cases):
    cases_text = "\n\n".join(
        f"CASE {i+1}:\n{case['text']}" for i, case in enumerate(similar_cases)
    )
    return f"""
You are a legal AI trained on Indian case law. Use Chain of Thought reasoning and reflect on your answer.

USER CASE: {user_case}

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
    messages = [
        {"role": "user", "content": prompt}
    ]
    first_response = chat_call(messages)

    reflexion_prompt = f"""
Your previous response was:

{first_response}

Reflect on this answer. Are there missing steps, incorrect assumptions, or stronger arguments to add? Revise and improve the analysis with better legal logic and structure.
""".strip()

    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user", "content": reflexion_prompt})

    revised_response = chat_call(messages)
    return first_response, revised_response

# === Main interactive loop ===
if __name__ == "__main__":
    print("\n⚖️ Legal Advisor – CoT + Reflexion (Groq LLaMA 3 70B)\n")

    while True:
        user_input = input("📥 Describe your legal case (or type 'exit'): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting Legal Advisor.")
            break

        print("🔍 Searching similar cases...")
        top_cases = search_faiss(user_input, top_k=5)
        print('top_cases = ', top_cases)

        print("🧠 Reasoning with CoT + Reflexion...")
        prompt = build_prompt(user_input, top_cases)
        original, improved = run_reflexion(prompt)

        print("\n🧩 ORIGINAL RESPONSE (Chain of Thought):\n")
        print(original)
        print("\n🔁 REFLECTED & IMPROVED LEGAL ADVICE:\n")
        print(improved)
        print("\n" + "=" * 100 + "\n")


#fastapi  se backend bnana h



# I was terminated by my company without any preceding notice, even when i had the notice period of 15 days, in my NDA. What are my options in this situation?
