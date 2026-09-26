import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

# Load environment variables
load_dotenv()

# -----------------------------
# 1. Connect to Pinecone
# -----------------------------

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-qa-index")

# -----------------------------
# 2. Load embedding model
# -----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 3. Connect to Gemini
# -----------------------------

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------
# 4. User question
# -----------------------------

question = input("Enter your question: ")

# Convert question into vector
query_vector = embedding_model.encode(question).tolist()

# -----------------------------
# 5. Search Pinecone
# -----------------------------

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

# -----------------------------
# 6. Prepare relevant context
# -----------------------------

context = ""

for match in results["matches"]:
    context += match["metadata"]["text"] + "\n\n"

# -----------------------------
# 7. Send context to Gemini
# -----------------------------

prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer is not available in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}
"""

response = gemini.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)

# -----------------------------
# 8. Display final answer
# -----------------------------

print("\nFinal Answer:")
print(response.output_text)