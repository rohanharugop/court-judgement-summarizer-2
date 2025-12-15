import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone

# =========================
# Load environment
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("❌ PINECONE_API_KEY or PINECONE_INDEX missing")

# =========================
# Init Pinecone
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# =========================
# Load chunks (UTF-8 safe)
# =========================
with open(
    "processed-data/judgment_chunks.json",
    encoding="utf-8",
    errors="ignore"
) as f:
    chunks = json.load(f)

print(f"📦 Loaded {len(chunks)} chunks")

# =========================
# Upload in batches
# =========================
BATCH_SIZE = 50
vectors = []

for i, chunk in enumerate(chunks):
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[chunk["text"]],
        parameters={"input_type": "passage"}
    )[0]["values"]

    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {
            "case_name": chunk["metadata"]["case_name"],
            "text": chunk["text"]
        }
    })

    if len(vectors) == BATCH_SIZE:
        index.upsert(vectors=vectors)
        print(f"⬆️ Uploaded {i + 1} vectors")
        vectors = []

# Upload remainder
if vectors:
    index.upsert(vectors=vectors)
    print(f"⬆️ Uploaded final batch")

print("✅ All vectors uploaded to Pinecone successfully")
