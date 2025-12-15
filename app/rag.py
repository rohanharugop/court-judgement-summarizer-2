import os
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq

# ✅ LOAD ENV FIRST
load_dotenv()

# =========================
# Init Clients
# =========================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# Retrieve Precedents
# =========================
def retrieve_precedents(query: str, top_k: int = 5):
    # Pinecone-hosted embedding
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )[0]["values"]

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    precedents = []
    for match in results["matches"]:
        precedents.append({
            "case_name": match["metadata"].get("case_name", "Unknown"),
            "excerpt": match["metadata"].get("text", "")[:800]
        })

    return precedents

# =========================
# Generate Explanation
# =========================
def generate_explanation(query: str, precedents: list):
    context = "\n\n".join(
        f"{p['case_name']}:\n{p['excerpt']}"
        for p in precedents
    )

    system_prompt = """
You are a legal research assistant.

• If the query is casual or greeting → respond politely and ask for a legal query.
• If the query is legal → explain using ONLY the provided precedents.
• Do NOT hallucinate cases.
• Keep response structured and concise.
• Answer with markdown formatting, including bullet points and headings where appropriate.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Query:
{query}

Precedents:
{context}
"""
            }
        ]
    )

    return response.choices[0].message.content
