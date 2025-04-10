import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# =====================
# Step 1: Load Data and Build Vector Store
# =====================
print("📄 Loading embeddings CSV...")
df = pd.read_csv("all_pdf_embeddings.csv")

dim_cols = sorted([col for col in df.columns if re.match(r"dim_\\d+", col)], key=lambda x: int(x.split("_")[1]))
vectors = df[dim_cols].values.astype("float32")
texts = df["text"].tolist()

print("📦 Building FAISS index...")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Save optional
faiss.write_index(index, "pdf_index.faiss")
with open("pdf_chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

# =====================
# Step 2: Load Models
# =====================
print("🤖 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("🤖 Loading Qwen1.5-0.5B...")
llm_model_id = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# =====================
# Step 3: RAG Functions
# =====================
def retrieve_similar_chunks(query, top_k=5):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [texts[i] for i in indices[0]]

def generate_with_qwen(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""
    response = generate_with_qwen(prompt)
    return response.split("Answer:")[-1].strip()

# =====================
# Step 4: Ask Questions
# =====================
if __name__ == "__main__":
    print("💬 RAG is ready. Ask your question (type 'exit' to quit):\n")

    while True:
        user_query = input("🧠 You: ")
        if user_query.lower() == "exit":
            break

        top_chunks = retrieve_similar_chunks(user_query)
        response = generate_answer(user_query, top_chunks)

        print("🤖 Qwen:", response)
        print("\n" + "-" * 60 + "\n")





