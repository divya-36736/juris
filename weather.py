import os
import json
import faiss
import gdown
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# === CONFIG ===
FAISS_INDEX_PATH = "faiss_legal.index"
METADATA_PATH = "faiss_legal_metadata.json"
EMBED_MODEL_PATH = "all-MiniLM-L6-v2"
GROQ_API_KEY = "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"
LLAMA_MODEL = "llama3-70b-8192"
TOP_K = 5

# === Drive URLs ===
FAISS_DRIVE_URL = "https://drive.google.com/uc?id=19NAfrrd6xsXukhepTunUkGe8oB1rWpJ6"
META_DRIVE_URL = "https://drive.google.com/uc?id=19NAzfK2-CNMLDWFLRspzrcHm-uly4YB-"
EMBED_DRIVE_FOLDER = "https://drive.google.com/drive/folders/19TU9YXiYgHAWXbGnya8OSsqJ6x3VNsm0"

# === Helper: Download from Drive if missing ===
def ensure_files():
    if not os.path.exists(FAISS_INDEX_PATH):
        print("⬇️ Downloading FAISS index...")
        gdown.download(FAISS_DRIVE_URL, FAISS_INDEX_PATH, quiet=False)

    if not os.path.exists(METADATA_PATH):
        print("⬇️ Downloading metadata...")
        gdown.download(META_DRIVE_URL, METADATA_PATH, quiet=False)

    model_files = [
        os.path.join(EMBED_MODEL_PATH, "config.json"),
        os.path.join(EMBED_MODEL_PATH, "pytorch_model.bin"),
        os.path.join(EMBED_MODEL_PATH, "sentence_bert_config.json"),
    ]
    if not all(os.path.exists(f) for f in model_files):
        print("⬇️ Embedding model not found. Please manually download from:")
        print(f"👉 {EMBED_DRIVE_FOLDER}")
        print(f"➡️ Unzip into folder: {EMBED_MODEL_PATH}")
        raise SystemExit("❌ Cannot proceed without embedding model.")


# === Prepare Files ===
ensure_files()

# === Load FAISS + Metadata ===
print("📦 Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL_PATH)
client = Groq(api_key=GROQ_API_KEY)

# === Core Functions ===
def search_faiss(query, top_k=TOP_K):
    embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        meta = metadata[idx]
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "file": meta.get("file", "Unknown"),
            "text": meta.get("text", "")[:1000],
            "case_name": meta.get("case_name", "Unknown Case"),
            "year": meta.get("year", "Unknown Year"),
            "verdict": meta.get("verdict", "No verdict info")
        })
    return results

def build_prompt(user_case, similar_cases):
    cases_text = "\n\n".join(
        f"CASE {i+1}:\n"
        f"Name: {case['case_name']}\n"
        f"Year: {case['year']}\n"
        f"Verdict: {case['verdict']}\n"
        f"Summary: {case['text']}"
        for i, case in enumerate(similar_cases)
    )

    return f"""
You are a legal AI trained on Indian case law. Use Chain of Thought reasoning and reflect on your answer.

USER CASE:
{user_case}

SIMILAR PRECEDENT CASES:
{cases_text}

TASK:
1. Analyze the user case facts.
2. Compare with precedents step-by-step.
3. Use Chain of Thought reasoning to assess strength.
4. Predict outcomes based on legal patterns and verdicts.
5. Reflect and revise for better logic.
6. Provide final structured legal advice with legal citations if helpful.
""".strip()

def chat_call(prompt):
    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1024,
        top_p=1,
        stop=None
    )
    return response.choices[0].message.content.strip()

def run_reflexion(prompt):
    first_response = chat_call(prompt)
    reflexion_prompt = f"""
Your previous response was:

{first_response}

Reflect on this answer. Are there missing steps, incorrect assumptions, or stronger arguments to add?
Revise and improve the analysis with better legal logic and structure.
""".strip()

    revised_response = chat_call(reflexion_prompt)
    return revised_response

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return ""

# === MAIN (For Local Testing) ===
if __name__ == "__main__":
    print("\n⚖️ Legal Advisor (FAISS + CoT + Reflexion + Groq LLaMA)\n")

    while True:
        choice = input("📤 Input via (1) Text or (2) PDF? (type 'exit' to quit): ").strip()
        if choice.lower() == 'exit':
            print("👋 Exiting.")
            break

        if choice == "1":
            user_input = input("📝 Enter your legal case details: ").strip()
        elif choice == "2":
            pdf_path = input("📄 Enter full PDF path: ").strip()
            user_input = extract_text_from_pdf(pdf_path)
            print("📖 Extracted text from PDF.")
        else:
            print("⚠️ Invalid choice. Try again.")
            continue

        print("🔍 Searching similar cases...")
        top_cases = search_faiss(user_input)

        print("🧠 Reasoning with CoT + Reflexion...")
        prompt = build_prompt(user_input, top_cases)
        final_response = run_reflexion(prompt)

        print("\n🔎 FINAL LEGAL ADVICE (Reflected):\n")
        print(final_response)
        print("\n" + "=" * 100 + "\n")



