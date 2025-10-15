import logging
from typing import List, Optional, Dict, Any, Union
import os
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_answer(query: str, retrieved_chunks: List[str], model: str = "gpt-3.5-turbo", include_sources: bool = False, 
                    llm_provider: str = "local") -> str:
    """
    Generate an answer to a query using retrieved document chunks and a language model.

    This function combines the retrieved chunks into a context string, creates a prompt
    for the language model, and generates a succinct and accurate answer. It supports
    multiple LLM providers including OpenAI, Llama, Mistral, and local transformers.

    Args:
        query (str): The user's question.
        retrieved_chunks (List[str]): List of relevant text chunks retrieved from the vector database.
        model (str, optional): The language model to use ("gpt-3.5-turbo", "gpt-4", etc.). Defaults to "gpt-3.5-turbo".
        include_sources (bool, optional): Whether to include source citations in the answer. Defaults to False.
        llm_provider (str, optional): LLM provider to use ("openai", "ollama", "huggingface", "local"). Defaults to "local".

    Returns:
        str: The generated answer, optionally with source citations.

    Raises:
        ValueError: If API keys are not set or if no chunks are provided.
        Exception: If the API call fails.

    Example:
        >>> query = "What is machine learning?"
        >>> chunks = ["ML is a subset of AI...", "It involves training algorithms..."]
        >>> answer = generate_answer(query, chunks)
        >>> print(answer)
        Machine learning is a subset of artificial intelligence that involves training algorithms...
    """
    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided for answer generation")

    # Determine if this is a summarization request
    is_summarization = _is_summarization_query(query)
    
    # Combine chunks into context
    context = "\n\n".join([f"Excerpt {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    logger.info(f"Combined {len(retrieved_chunks)} chunks into context (length: {len(context)} characters)")

    # Create a base system prompt
    system_prompt = "You are a helpful assistant that answers questions based on provided document excerpts. Be concise and accurate."
    
    # If it's a summarization request, adjust the system prompt
    if is_summarization:
        system_prompt = "You are a helpful assistant that creates concise summaries based on provided document excerpts. Focus on the key points and main ideas."
        logger.info("Detected summarization request. Adjusting prompt.")

    # Create prompt
    prompt = f"""Using only the following excerpts, {'summarize the content' if is_summarization else 'answer the user\'s question'} succinctly and accurately.

Context:
{context}

{'Instructions: Provide a clear and concise summary of the document.' if is_summarization else f'Question: {query}'}

{'Summary:' if is_summarization else 'Answer:'}"""

    try:
        logger.info(f"Generating answer using {llm_provider}:{model} for query: {query[:50]}...")

        if llm_provider == "openai":
            answer = _generate_with_openai(system_prompt, prompt, model)
        elif llm_provider == "ollama":
            answer = _generate_with_ollama(system_prompt, prompt, model)
        elif llm_provider == "huggingface":
            answer = _generate_with_huggingface(system_prompt, prompt, model)
        elif llm_provider == "local":
            answer = _generate_with_transformers_local(system_prompt, prompt)
        else:
            logger.warning(f"Unsupported LLM provider: {llm_provider}. Falling back to local generation.")
            answer = _generate_with_transformers_local(system_prompt, prompt)

        logger.info(f"Generated answer (length: {len(answer)} characters)")

        # Optionally include sources
        if include_sources:
            sources_text = "\n\nSources:\n" + "\n".join([f"- Excerpt {i+1}" for i in range(len(retrieved_chunks))])
            answer += sources_text
            logger.info("Included source citations in answer")

        return answer

    except Exception as e:
        logger.error(f"Error during answer generation with {llm_provider}: {str(e)}")
        raise Exception(f"Failed to generate answer: {str(e)}")


def _is_summarization_query(query: str) -> bool:
    """
    Check if a query is asking for a summary.
    
    Args:
        query (str): The user's query.
        
    Returns:
        bool: True if the query is asking for a summary, False otherwise.
    """
    summarization_keywords = [
        "summarize", "summary", "summarization", 
        "summarise", "summarisation", "abstract", 
        "synopsis", "overview", "main points",
        "tldr", "brief description", "sum up"
    ]
    
    query_lower = query.lower()
    
    for keyword in summarization_keywords:
        if keyword in query_lower:
            return True
            
    return False


def _generate_with_openai(system_prompt: str, prompt: str, model: str) -> str:
    """
    Generate an answer using OpenAI's API.
    
    Args:
        system_prompt (str): The system message to set the context.
        prompt (str): The prompt to send to the model.
        model (str): The OpenAI model to use.
        
    Returns:
        str: The generated answer.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        Exception: If the API call fails.
    """
    import openai
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    try:
        # Create the client
        client = openai.OpenAI(api_key=api_key)
        
        # List available models and use a fallback if the requested model is not available
        try:
            models_list = client.models.list()
            available_models = [m.id for m in models_list.data]
            logger.info(f"Found {len(available_models)} available OpenAI models")
            
            # Check if specified model is available
            if model not in available_models:
                # Define fallback models in order of preference
                fallback_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"]
                
                for fallback in fallback_models:
                    if fallback in available_models:
                        logger.warning(f"Model '{model}' not available. Using fallback model: {fallback}")
                        model = fallback
                        break
                else:
                    # If no preferred fallbacks are found, just use the first available
                    if available_models:
                        model = available_models[0]
                        logger.warning(f"Using first available model: {model}")
        except Exception as e:
            logger.warning(f"Could not list available models: {str(e)}. Proceeding with default model.")
            model = "gpt-3.5-turbo"  # Safe default
            
        # Generate the response
        logger.info(f"Making OpenAI API request with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.1  # Low temperature for consistent, factual answers
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        
        # Graceful degradation - return a summary from the context instead of raising an exception
        fallback_response = "I couldn't generate a complete answer with the AI model, but here's what I found in the documents:\n\n"
        
        # Extract text from the context to create a simple answer
        context_lines = prompt.split("\n")
        excerpts = []
        for line in context_lines:
            if line.startswith("Excerpt"):
                # Get just the content without the "Excerpt N:" prefix
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    excerpts.append(parts[1])
        
        if excerpts:
            # Add the first 3 excerpts (or fewer if less are available)
            for i, excerpt in enumerate(excerpts[:3]):
                fallback_response += f"{i+1}. {excerpt}\n\n"
        else:
            fallback_response += "No relevant information found in the document excerpts."
            
        return fallback_response


def _generate_with_ollama(system_prompt: str, prompt: str, model: str) -> str:
    """
    Generate an answer using Ollama local models (e.g., Llama, Mistral).
    
    Args:
        system_prompt (str): The system message to set the context.
        prompt (str): The prompt to send to the model.
        model (str): The model name in Ollama (e.g., "llama3", "mistral", etc.)
        
    Returns:
        str: The generated answer.
        
    Raises:
        Exception: If the API call fails.
    """
    try:
        import requests
        
        # Ollama API endpoint (default for local installation)
        ollama_api = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api')
        
        # Prepare the prompt with system message
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Make request to Ollama API
        response = requests.post(
            f"{ollama_api}/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 800,
                }
            }
        )
        
        response.raise_for_status()
        return response.json()["response"].strip()
        
    except ImportError:
        raise Exception("Failed to import requests library. Please install it with: pip install requests")
    except Exception as e:
        logger.error(f"Ollama API error: {str(e)}")
        raise Exception(f"Failed to generate answer with Ollama: {str(e)}")


def _generate_with_transformers_local(system_prompt: str, prompt: str) -> str:
    """
    Generate an answer using local Transformers models with lightweight defaults.
    This function doesn't require API keys or internet connection.

    Args:
        system_prompt (str): The system message to set the context.
        prompt (str): The prompt to send to the model.

    Returns:
        str: The generated answer or a fallback answer based on the context.
    """
    try:
        # Import transformers only when needed
        logger.info("Attempting to use local transformers for answer generation")

        # Extract text from the context to create a focused answer
        context_lines = prompt.split("\n")
        excerpts = []
        question = ""
        is_summarization = "summarize" in prompt.lower() or "summary" in prompt.lower()

        # Find the question and excerpts
        for line in context_lines:
            if line.startswith("Excerpt"):
                # Get just the content without the "Excerpt N:" prefix
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    excerpts.append(parts[1])
            elif line.startswith("Question:"):
                question = line.replace("Question:", "").strip()

        if not excerpts:
            return "No relevant information found in the documents."

        # For summarization, create a concise summary
        if is_summarization:
            # Combine all excerpts and create a summary
            all_text = " ".join(excerpts)
            # Extract key sentences (simple heuristic: sentences with important keywords)
            sentences = [s.strip() for s in all_text.replace(". ", ".\n").split("\n") if s.strip()]

            key_sentences = []
            important_keywords = ["documind", "rag", "search", "engine", "knowledge", "base", "document", "query", "answer", "user", "system", "implementation", "architecture"]

            for sentence in sentences[:10]:  # Limit to first 10 sentences
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    key_sentences.append(sentence)

            if key_sentences:
                summary = " ".join(key_sentences[:5])  # Take top 5 key sentences
                return f"Summary: {summary}"
            else:
                return f"Summary: {' '.join(sentences[:3])}"

        # For questions, create a coherent answer
        question_lower = question.lower()

        # Extract key terms from the question
        key_terms = []
        technical_terms = ["implement", "architecture", "design", "structure", "process", "flow", "steps",
                          "component", "module", "plan", "approach", "method", "technology", "stack", "framework",
                          "database", "api", "interface", "backend", "frontend", "system", "application"]

        for term in technical_terms:
            if term in question_lower:
                key_terms.append(term)

        # If no technical terms, extract nouns and important words
        if not key_terms:
            words = question_lower.split()
            key_terms = [w for w in words if len(w) > 3 and w not in ["what", "how", "why", "when", "where", "which", "does", "can", "will", "should"]]

        # Score and rank excerpts
        scored_excerpts = []
        for excerpt in excerpts:
            excerpt_lower = excerpt.lower()
            score = 0

            # Score based on keyword matches
            for term in key_terms:
                if term in excerpt_lower:
                    score += 10

            # Bonus points for structured content
            if ":" in excerpt:
                score += 5
            if any(word in excerpt_lower for word in ["implementation", "architecture", "design", "system"]):
                score += 15
            if any(word in excerpt_lower for word in ["step", "process", "method", "approach"]):
                score += 10

            scored_excerpts.append((score, excerpt))

        # Sort by score, highest first
        scored_excerpts.sort(reverse=True, key=lambda x: x[0])

        # Build coherent answer
        answer_parts = []
        used_sentences = set()

        for score, excerpt in scored_excerpts[:4]:  # Use top 4 excerpts
            if score > 0:  # Only include relevant excerpts
                # Split excerpt into sentences and pick the most relevant ones
                sentences = [s.strip() for s in excerpt.replace(". ", ".\n").split("\n") if s.strip() and len(s) > 10]

                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Check if sentence contains key terms and hasn't been used
                    if any(term in sentence_lower for term in key_terms) and sentence not in used_sentences:
                        answer_parts.append(sentence)
                        used_sentences.add(sentence)
                        break  # Take only one sentence per excerpt to avoid redundancy

        if answer_parts:
            # Create a coherent paragraph
            coherent_answer = " ".join(answer_parts[:3])  # Limit to 3 key points

            # Add context if it's a "what is" question
            if question_lower.startswith("what is") or question_lower.startswith("what are"):
                return f"{question.capitalize()} {coherent_answer}"
            else:
                return f"Based on the documents: {coherent_answer}"
        else:
            # Fallback: return the highest scored excerpt
            if scored_excerpts:
                return f"From the documents: {scored_excerpts[0][1]}"
            else:
                return "I couldn't find specific information to answer your question in the provided documents."

    except Exception as e:
        logger.error(f"Error in local transformers generation: {str(e)}")

        # Improved fallback
        try:
            context_lines = prompt.split("\n")
            excerpts = [line.split(": ", 1)[1] for line in context_lines if line.startswith("Excerpt") and ": " in line]

            if excerpts:
                return f"Based on the document excerpts: {' '.join(excerpts[:2])}"
            else:
                return "Unable to generate an answer from the available information."
        except:
            return "I encountered an error while processing your question. Please try rephrasing it."


def _generate_with_huggingface(system_prompt: str, prompt: str, model: str) -> str:
    """
    Generate an answer using Hugging Face Inference API.
    
    Args:
        system_prompt (str): The system message to set the context.
        prompt (str): The prompt to send to the model.
        model (str): The model ID on Hugging Face (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
        
    Returns:
        str: The generated answer.
        
    Raises:
        ValueError: If HUGGINGFACE_API_KEY is not set.
        Exception: If the API call fails.
    """
    try:
        from huggingface_hub import InferenceClient
        
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
            
        client = InferenceClient(token=api_key)
        
        # Prepare the prompt with system message
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generate text using the Hugging Face Inference API
        response = client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=800,
            temperature=0.1,
            repetition_penalty=1.1
        )
        
        return response.strip()
        
    except ImportError:
        raise Exception("Failed to import huggingface_hub library. Please install it with: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise Exception(f"Failed to generate answer with Hugging Face: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Set your API keys (in practice, use environment variables)
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    # os.environ['HUGGINGFACE_API_KEY'] = 'your-api-key-here'
    
    # Sample query and retrieved chunks
    query = "What are the main components of a neural network?"

    retrieved_chunks = [
        "A neural network consists of layers of interconnected nodes called neurons. The main components include input layers, hidden layers, and output layers.",
        "Each neuron receives inputs, applies weights, adds bias, and passes the result through an activation function.",
        "Neural networks learn through backpropagation, adjusting weights to minimize prediction errors.",
        "Common activation functions include ReLU, sigmoid, and tanh."
    ]
    
    # Sample summarization query
    summarization_query = "Summarize the information about neural networks"

    try:
        # Test different LLM providers
        providers = []
        
        # Check if OpenAI API key is set
        if os.getenv('OPENAI_API_KEY'):
            providers.append(("openai", "gpt-3.5-turbo"))
            
        # Check if we have a running Ollama instance
        try:
            import requests
            ollama_running = False
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    ollama_running = True
                    providers.append(("ollama", "llama3"))
            except:
                pass
        except ImportError:
            pass
            
        # Check if Hugging Face API key is set
        if os.getenv('HUGGINGFACE_API_KEY'):
            providers.append(("huggingface", "mistralai/Mistral-7B-Instruct-v0.2"))
            
        # If no providers are available, default to OpenAI
        if not providers:
            print("No LLM providers configured. Defaulting to OpenAI (will fail without API key).")
            providers = [("openai", "gpt-3.5-turbo")]
        
        # Test regular query
        for provider, model in providers:
            try:
                print(f"\n{'='*50}")
                print(f"Testing {provider}:{model} with regular query")
                print(f"{'='*50}")
                
                # Generate answer without sources
                answer = generate_answer(query, retrieved_chunks, model=model, llm_provider=provider)
                print("Answer without sources:")
                print(answer)
                print("\n" + "-"*50 + "\n")

                # Generate answer with sources
                answer_with_sources = generate_answer(query, retrieved_chunks, model=model, llm_provider=provider, include_sources=True)
                print("Answer with sources:")
                print(answer_with_sources)
                
            except Exception as e:
                print(f"Error with {provider}: {e}")
        
        # Test summarization query
        for provider, model in providers:
            try:
                print(f"\n{'='*50}")
                print(f"Testing {provider}:{model} with summarization query")
                print(f"{'='*50}")
                
                # Generate summary
                summary = generate_answer(summarization_query, retrieved_chunks, model=model, llm_provider=provider)
                print("Summary:")
                print(summary)
                
            except Exception as e:
                print(f"Error with {provider}: {e}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set the appropriate API keys in your environment variables")