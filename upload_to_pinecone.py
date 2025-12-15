from pinecone import Pinecone
import json, os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-rag-index")

with open("processed-data/judgment_chunks.json") as f:
    chunks = json.load(f)

vectors = []
for i, chunk in enumerate(chunks):
    emb = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[chunk["text"]],
        parameters={"input_type": "passage"}
    )[0]["values"]

    vectors.append({
        "id": str(i),
        "values": emb,
        "metadata": {
            "case_name": chunk["metadata"]["case_name"],
            "text": chunk["text"]
        }
    })

index.upsert(vectors)
