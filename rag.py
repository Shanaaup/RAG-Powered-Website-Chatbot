import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "scraped_data"
CHUNK_SIZE = 200
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# Load text
def load_documents(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


# Chunk text
def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    docs = load_documents(DATA_FOLDER)

    if not docs:
        print("No .txt files found in scraped_data folder!")
        return

    print("Chunking...")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc, CHUNK_SIZE))

    print("Creating embeddings...")
    embeddings = model.encode(chunks)

    while True:
        query = input("\nAsk question (or 'exit'): ")
        if query.lower() == "exit":
            break

        query_embedding = model.encode([query])[0]

        scores = []
        for emb in embeddings:
            score = cosine_similarity(query_embedding, emb)
            scores.append(score)

        top_index = np.argmax(scores)

        print("\n🔎 Most Relevant Chunk:\n")
        print(chunks[top_index][:1000])


if __name__ == "__main__":
    main()