import os
import pickle

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-qa-index")

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "What was the objective of the internship?"

query_vector = model.encode(question).tolist()

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print("\nTop relevant results:\n")

for match in results["matches"]:
    print("Score:", match["score"])
    print("Text:", match["metadata"]["text"])
    print("-" * 60)
    