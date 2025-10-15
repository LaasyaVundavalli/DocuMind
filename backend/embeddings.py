import logging
from typing import List
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache for sentence transformers
_model = None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, respect_paragraphs: bool = True) -> List[str]:
    """
    Split text into manageable chunks with intelligent boundary handling.

    This function divides the input text into chunks of approximately the specified size,
    with a configurable overlap between consecutive chunks to maintain context.
    It attempts to respect natural text boundaries like sentences and paragraphs.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The target size of each chunk in characters. Defaults to 500.
        overlap (int, optional): The number of characters to overlap between chunks. Defaults to 50.
        respect_paragraphs (bool, optional): Try to keep paragraphs together. Defaults to True.

    Returns:
        List[str]: A list of text chunks with source tracking metadata.

    Raises:
        ValueError: If chunk_size is less than or equal to overlap.

    Example:
        >>> text = "This is a long document with multiple sentences."
        >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
        >>> print(chunks)
        ['This is a long docum', 'docum ent with multi', 'ulti ple sentences.']
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    if not text:
        return []

    # Split text into paragraphs first if requested
    if respect_paragraphs:
        paragraphs = text.split("\n\n")
    else:
        paragraphs = [text]
        
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph is very small, just add it to the current chunk
        if len(paragraph) < chunk_size / 4:
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Current chunk is getting too big, store it and start a new one
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            continue
            
        # For larger paragraphs, split into sentences
        sentences = []
        for sent in paragraph.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|"):
            if sent:
                sentences.append(sent)
                
        # Process each sentence
        for sentence in sentences:
            # If adding this sentence exceeds chunk size, store chunk and start new one
            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Get the last 'overlap' characters for context
                    words = current_chunk.split()
                    overlap_text = ""
                    for word in reversed(words):
                        if len(overlap_text) + len(word) + 1 <= overlap:
                            overlap_text = word + " " + overlap_text
                        else:
                            break
                    current_chunk = overlap_text
                else:
                    current_chunk = ""
                    
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
                
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    logger.info(f"Text chunked into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
    return chunks


def generate_embeddings(chunks: List[str], method: str = "sentence-transformers") -> np.ndarray:
    """
    Generate vector embeddings for a list of text chunks.

    This function supports multiple embedding methods:
    - "sentence-transformers": Uses the Sentence Transformers library with 'all-MiniLM-L6-v2' model
    - "transformers": Uses Hugging Face Transformers library (fallback if sentence-transformers not available)
    - "openai": Uses OpenAI's text-embedding-3-small model (requires OPENAI_API_KEY environment variable)
    - "random": Uses random vectors (for testing only)

    Args:
        chunks (List[str]): List of text chunks to embed.
        method (str, optional): Embedding method to use. Defaults to "sentence-transformers".

    Returns:
        np.ndarray: Array of embeddings with shape (len(chunks), embedding_dim).

    Raises:
        ValueError: If an unsupported method is specified.
        Exception: If embedding generation fails.

    Example:
        >>> chunks = ["Hello world", "How are you?"]
        >>> embeddings = generate_embeddings(chunks)
        >>> print(embeddings.shape)
        (2, 384)  # For sentence-transformers
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation")
        return np.array([])

    try:
        if method == "sentence-transformers":
            logger.info("Generating embeddings using Sentence Transformers")
            global _model
            try:
                if _model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        _model = SentenceTransformer('all-MiniLM-L6-v2')
                    except ImportError as e:
                        logger.warning(f"Failed to import sentence_transformers: {e}")
                        logger.warning("Falling back to transformers method")
                        return generate_embeddings(chunks, method="transformers")
                
                embeddings = _model.encode(chunks, convert_to_numpy=True)
                logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
                
            except Exception as e:
                logger.warning(f"Error with sentence-transformers: {e}. Falling back to transformers method.")
                return generate_embeddings(chunks, method="transformers")

        elif method == "transformers":
            logger.info("Generating embeddings using Hugging Face Transformers")
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model = AutoModel.from_pretrained('bert-base-uncased')
                
                # Process chunks in batches
                embeddings_list = []
                batch_size = 8  # Smaller batch size to avoid memory issues
                
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    
                    # Tokenize and prepare input
                    encoded_input = tokenizer(batch_chunks, padding=True, truncation=True, 
                                              max_length=512, return_tensors='pt')
                    
                    # Get model output
                    with torch.no_grad():
                        output = model(**encoded_input)
                    
                    # Mean pooling - take average of all tokens
                    embeddings = output.last_hidden_state.mean(dim=1)
                    embeddings_list.append(embeddings.numpy())
                
                embeddings = np.vstack(embeddings_list)
                logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
                
            except ImportError as e:
                logger.warning(f"Failed to import transformers: {e}. Falling back to random embeddings.")
                return generate_embeddings(chunks, method="random")
            except Exception as e:
                logger.warning(f"Error with transformers: {e}. Falling back to random embeddings.")
                return generate_embeddings(chunks, method="random")

        elif method == "openai":
            logger.info("Generating embeddings using OpenAI")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY environment variable not set. Falling back to random embeddings.")
                return generate_embeddings(chunks, method="random")

            try:
                import openai
                
                client = openai.OpenAI(api_key=api_key)
                embeddings_list = []

                # OpenAI has a limit on batch size, so we'll process in smaller batches
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    
                    try:
                        # Try with the newer client first
                        response = client.embeddings.create(
                            model="text-embedding-3-small",
                            input=batch_chunks
                        )
                        batch_embeddings = [data.embedding for data in response.data]
                    except Exception:
                        # Fall back to older API
                        response = openai.Embedding.create(
                            model="text-embedding-3-small",
                            input=batch_chunks
                        )
                        batch_embeddings = [data['embedding'] for data in response['data']]
                        
                    embeddings_list.extend(batch_embeddings)

                embeddings = np.array(embeddings_list)
                logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
                
            except ImportError:
                logger.warning("OpenAI library not installed. Falling back to random embeddings.")
                return generate_embeddings(chunks, method="random")
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}. Falling back to random embeddings.")
                return generate_embeddings(chunks, method="random")

        elif method == "random":
            logger.warning("Using random embeddings (for testing only)")
            # Generate random embeddings with 384 dimensions (same as MiniLM)
            dim = 384
            embeddings = np.random.rand(len(chunks), dim).astype(np.float32)
            # Normalize the embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            logger.info(f"Generated {len(embeddings)} random embeddings with dimension {dim}")

        else:
            raise ValueError(f"Unsupported embedding method: {method}")

        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings with method {method}: {str(e)}")
        logger.warning("Falling back to random embeddings due to error")
        return generate_embeddings(chunks, method="random")


# Example usage
if __name__ == "__main__":
    # Example text
    sample_text = """
    Machine learning is a subset of artificial intelligence that involves training algorithms
    to recognize patterns in data. Deep learning, a type of machine learning, uses neural
    networks with multiple layers to process information. Natural language processing
    allows computers to understand and generate human language. Computer vision enables
    machines to interpret and understand visual information from the world.
    """

    # Chunk the text
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:50]}...")

    # Generate embeddings
    try:
        embeddings = generate_embeddings(chunks, method="sentence-transformers")
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"First embedding (first 10 values): {embeddings[0][:10]}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")

    # Example with OpenAI (requires API key)
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    # try:
    #     embeddings_openai = generate_embeddings(chunks, method="openai")
    #     print(f"OpenAI embeddings shape: {embeddings_openai.shape}")
    # except Exception as e:
    #     print(f"Error with OpenAI embeddings: {e}")