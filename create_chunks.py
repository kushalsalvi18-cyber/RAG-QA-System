from langchain_text_splitters import RecursiveCharacterTextSplitter

# OCR se nikla hua text
input_file = "documents/industrial_training_ocr.txt"

with open(input_file, "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))

# Text ko chunks me divide karna
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([text])

print("Total chunks:", len(chunks))

# Chunks ko save karna
output_file = "documents/industrial_training_chunks.txt"

with open(output_file, "w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks):
        file.write(f"\n\n===== CHUNK {i} =====\n\n")
        file.write(chunk.page_content)

print("Chunks saved successfully!")
print("Saved file:", output_file)