import pickle
from sentence_transformers import SentenceTransformer

# Chunks file read karo
input_file = "documents/industrial_training_chunks.txt"

with open(input_file, "r", encoding="utf-8") as file:
    text = file.read()

# Har chunk ko alag karo
chunks = text.split("===== CHUNK ")

# Empty entries hatao
chunks = [chunk for chunk in chunks if chunk.strip()]

print("Total chunks:", len(chunks))

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings generate karo
embeddings = []

for i, chunk in enumerate(chunks):
    print(f"Creating embedding: {i + 1}/{len(chunks)}")

    vector = model.encode(chunk).tolist()

    embeddings.append(vector)

print("\nEmbeddings created successfully!")
print("Total embeddings:", len(embeddings))
print("Vector dimensions:", len(embeddings[0]))

# Chunks + embeddings save karo
data = {
    "chunks": chunks,
    "embeddings": embeddings
}

output_file = "documents/embeddings.pkl"

with open(output_file, "wb") as file:
    pickle.dump(data, file)

print("Embeddings saved successfully!")
print("Saved file:", output_file)