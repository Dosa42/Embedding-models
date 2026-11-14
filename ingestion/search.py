"""Search a native ChromaDB using the original model and tokenizer."""
import argparse
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from .core import Embedder


def search(database, model_dir, query, top_k=5):
    if not 1 <= top_k <= 100:
        raise ValueError('top_k must be 1..100')
    if not Path(database).is_dir():
        raise ValueError('Pass the complete chroma_db directory')
    client = chromadb.PersistentClient(path=str(database), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection('chunks', embedding_function=None)
        metadata = json.loads(collection.metadata['provenance_json'])
        embedder = Embedder(model_dir, metadata['model'].split('/')[-1], metadata['dtype'])
        for key in ('weights_sha256', 'tokenizer_sha256', 'dimension', 'pooling'):
            if embedder.provenance[key] != metadata[key]:
                raise ValueError('Query model differs from database: ' + key)
        vector = embedder.encode([query], query=True)[0]
        result = collection.query(query_embeddings=[vector.tolist()], n_results=min(top_k, collection.count()), include=['documents', 'metadatas', 'distances'])
        return [{'score': 1.0 - distance, 'id': identity, 'source': meta['source'], 'page': meta.get('page'), 'segment': meta['segment'], 'text': text}
                for identity, text, meta, distance in zip(result['ids'][0], result['documents'][0], result['metadatas'][0], result['distances'][0])]
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, help='Path to the extracted chroma_db directory')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(search(args.database, args.model_dir, args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
