import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.HttpClient(host="localhost", port=8000)


#Collection Creation
collection = chroma_client.get_or_create_collection(name="amazon_entities")

config = {
    "data_path": "C:\\Users\\KIIT0001\\Desktop\\amazon\\train_40k.csv",
    "kg_output_path": "output/amazon_kg.json",
    "vectdb_output_path": "output/amazon_embeddings.json",
}

# To Ensure output directories exist
os.makedirs(os.path.dirname(config["kg_output_path"]), exist_ok=True)
os.makedirs(os.path.dirname(config["vectdb_output_path"]), exist_ok=True)

# Load dataset
df = pd.read_csv(config["data_path"])
records = df.to_dict(orient="records")

# Build KG triples
triples = []
entities = set()

for row in tqdm(records):
    pid = row["Id"]
    cat1 = row["Cat1"]
    cat2 = row["Cat2"]
    cat3 = row["Cat3"]

    triples.extend([
        (pid, "belongs_to", cat1),
        (cat1, "subclass_of", cat2),
        (cat2, "subclass_of", cat3),
    ])
    entities.update([pid, cat1, cat2, cat3])

# Generating dummy embeddings
embedding_dim = 128
embeddings = {ent: np.random.rand(embedding_dim).tolist() for ent in entities}

# Add to Chroma
ids = list(embeddings.keys())
collection.add(
    ids=ids,
    embeddings=[embeddings[i] for i in ids],
    documents=ids
)

# Save KG triples
with open(config["kg_output_path"], "w") as f:
    json.dump([{"head": h, "relation": r, "tail": t} for (h, r, t) in triples], f, indent=2)

# Save vector embeddings
with open(config["vectdb_output_path"], "w") as f:
    json.dump(embeddings, f, indent=2)

# test query
query_vector = np.random.rand(embedding_dim).tolist()
results = collection.query(query_embeddings=[query_vector], n_results=5)
print("Sample vector DB results:", results)

print(f"✅ Knowledge Graph saved to {config['kg_output_path']}")
print(f"✅ Embeddings saved to {config['vectdb_output_path']}")
