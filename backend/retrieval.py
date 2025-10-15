import os
import logging
from typing import List, Tuple, Union, Any
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_vector_db(db_type: str = "faiss", db_path: str = "data/vector_store/") -> Any:
    """
    Initialize a vector database with persistence support.

    Supports FAISS and ChromaDB. Creates the database if it doesn't exist,
    or loads an existing one from disk.

    Args:
        db_type (str, optional): Type of vector database ("faiss" or "chromadb"). Defaults to "faiss".
        db_path (str, optional): Path to store/load the database. Defaults to "data/vector_store/".

    Returns:
        Any: The initialized vector database object (FAISS index or ChromaDB collection).

    Raises:
        ValueError: If an unsupported database type is specified.

    Example:
        >>> db = initialize_vector_db("faiss", "data/vector_store/")
        >>> print(type(db))
        <class 'faiss.swigfaiss.IndexFlatIP'>
    """
    os.makedirs(db_path, exist_ok=True)

    if db_type == "faiss":
        index_path = os.path.join(db_path, "faiss_index.idx")

        if os.path.exists(index_path):
            # Load existing index
            index = faiss.read_index(index_path)
            logger.info(f"Loaded existing FAISS index from {index_path}")
        else:
            # Create new index (will be initialized when first embeddings are added)
            index = None
            logger.info(f"FAISS index will be created when embeddings are added")

        return {"type": "faiss", "index": index, "path": index_path, "chunks": []}

    elif db_type == "chromadb":
        client_path = os.path.join(db_path, "chroma_db")

        client = chromadb.PersistentClient(path=client_path)
        collection = client.get_or_create_collection(name="documents")

        logger.info(f"Initialized ChromaDB collection at {client_path}")
        return {"type": "chromadb", "client": client, "collection": collection}

    else:
        raise ValueError(f"Unsupported database type: {db_type}. Use 'faiss' or 'chromadb'")


def add_embeddings_to_db(chunks: List[str], embeddings: np.ndarray, db: Any) -> None:
    """
    Add embeddings and their corresponding chunks to the vector database.

    Args:
        chunks (List[str]): List of text chunks corresponding to the embeddings.
        embeddings (np.ndarray): Array of embeddings with shape (len(chunks), embedding_dim).
        db (Any): The vector database object returned by initialize_vector_db.

    Raises:
        ValueError: If embeddings and chunks have mismatched lengths or invalid database type.

    Example:
        >>> chunks = ["chunk1", "chunk2"]
        >>> embeddings = np.random.rand(2, 384)
        >>> db = initialize_vector_db()
        >>> add_embeddings_to_db(chunks, embeddings, db)
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Number of chunks must match number of embeddings")

    if db["type"] == "faiss":
        dimension = embeddings.shape[1]

        if db["index"] is None:
            # Create new index
            db["index"] = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logger.info(f"Created new FAISS index with dimension {dimension}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        db["index"].add(embeddings.astype(np.float32))
        db["chunks"].extend(chunks)

        # Save index to disk
        faiss.write_index(db["index"], db["path"])
        logger.info(f"Added {len(chunks)} embeddings to FAISS index and saved to {db['path']}")

    elif db["type"] == "chromadb":
        # ChromaDB handles persistence automatically
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        db["collection"].add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} embeddings to ChromaDB collection")

    else:
        raise ValueError(f"Invalid database type: {db.get('type')}")


def query_vector_db(query_embedding: np.ndarray, db: Any, top_k: int = 5, return_scores: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Query the vector database for the most similar chunks.

    Args:
        query_embedding (np.ndarray): The query embedding vector.
        db (Any): The vector database object.
        top_k (int, optional): Number of top similar results to return. Defaults to 5.
        return_scores (bool, optional): Whether to return similarity scores along with chunks. Defaults to False.

    Returns:
        Union[List[str], List[Tuple[str, float]]]: List of top-k chunks, or list of (chunk, score) tuples if return_scores=True.

    Raises:
        ValueError: If database is empty or invalid.

    Example:
        >>> query_emb = np.random.rand(384)
        >>> results = query_vector_db(query_emb, db, top_k=3)
        >>> print(results)
        ['relevant chunk 1', 'relevant chunk 2', 'relevant chunk 3']
    """
    if db["type"] == "faiss":
        if db["index"] is None or db["index"].ntotal == 0:
            raise ValueError("Vector database is empty. Add embeddings first.")

        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search for top-k similar vectors
        scores, indices = db["index"].search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(db["chunks"]):  # Safety check
                chunk = db["chunks"][idx]
                if return_scores:
                    results.append((chunk, float(scores[0][i])))
                else:
                    results.append(chunk)

        logger.info(f"FAISS query returned {len(results)} results")

    elif db["type"] == "chromadb":
        # ChromaDB query
        query_result = db["collection"].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        results = []
        documents = query_result.get("documents", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        for i, doc in enumerate(documents):
            if return_scores:
                # Convert distance to similarity score (ChromaDB returns distances)
                score = 1.0 / (1.0 + distances[i]) if distances else 0.0
                results.append((doc, score))
            else:
                results.append(doc)

        logger.info(f"ChromaDB query returned {len(results)} results")

    else:
        raise ValueError(f"Invalid database type: {db.get('type')}")

    return results


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = initialize_vector_db("faiss", "data/vector_store/")

    # Sample data
    sample_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information."
    ]

    # Generate sample embeddings (normally from embeddings.py)
    np.random.seed(42)
    sample_embeddings = np.random.rand(len(sample_chunks), 384)  # Assuming 384-dim embeddings

    # Add to database
    add_embeddings_to_db(sample_chunks, sample_embeddings, db)

    # Query the database
    query_emb = np.random.rand(384)
    results = query_vector_db(query_emb, db, top_k=2)
    print("Top 2 results:")
    for i, chunk in enumerate(results):
        print(f"{i+1}. {chunk}")

    # Query with scores
    results_with_scores = query_vector_db(query_emb, db, top_k=2, return_scores=True)
    print("\nTop 2 results with scores:")
    for i, (chunk, score) in enumerate(results_with_scores):
        print(f"{i+1}. {chunk} (score: {score:.4f})")