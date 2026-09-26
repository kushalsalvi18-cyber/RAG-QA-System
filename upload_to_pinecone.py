import os
import pickle

from dotenv import load_dotenv
from pinecone import Pinecone


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")


# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("rag-qa-index")


# Load saved chunks and embeddings
with open("documents/embeddings.pkl", "rb") as file:
    data = pickle.load(file)

chunks = data["chunks"]
embeddings = data["embeddings"]

print("Total chunks:", len(chunks))
print("Vector dimensions:", len(embeddings[0]))


# Prepare records
records = []

for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):

    records.append({
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": {
            "text": chunk
        }
    })


# Upload vectors to Pinecone
index.upsert(vectors=records)

print("Upload completed successfully!")
print("Total vectors uploaded:", len(records))