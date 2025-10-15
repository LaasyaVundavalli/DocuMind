import streamlit as st
import requests
from typing import List
import time
import json

# Configure page
st.set_page_config(
    page_title="📚 DocuMind",
    layout="wide",
    page_icon="📚"
)

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Custom CSS for modern styling
st.markdown("""
<style>
/* Main Header */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.3rem;
}

/* Caption / Subtitle */
.caption-text {
    text-align: center;
    color: #6c757d;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

/* Card sections */
.card {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid #e0e0e0;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}

/* Results section */
.results-card {
    background-color: #e8f4f8;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 6px solid #1f77b4;
    margin-top: 2rem;
}

/* Success and Error */
.success-message { color: #28a745; font-weight: bold; }
.error-message { color: #dc3545; font-weight: bold; }

/* Buttons */
.stButton>button {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #155a8a;
}
</style>
""", unsafe_allow_html=True)


def upload_files_to_backend(files: List) -> dict:
    files_data = [("files", (file.name, file.getvalue(), file.type)) for file in files]
    try:
        response = requests.post(f"{BACKEND_URL}/upload", files=files_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def upload_text_to_backend(text: str) -> dict:
    files_data = [("files", ("pasted_text.txt", text, "text/plain"))]
    try:
        response = requests.post(f"{BACKEND_URL}/upload", files=files_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def query_backend(query: str, include_sources: bool = False, llm_provider: str = "openai", model: str = "gpt-4") -> dict:
    """
    Send a query to the backend API.

    Args:
        query: The user's question.
        include_sources: Whether to include source citations.
        llm_provider: The LLM provider to use (openai, ollama, huggingface).
        model: The specific model to use.

    Returns:
        dict: Response from the backend API.
    """
    payload = {
        "query": query,
        "include_sources": include_sources,
        "llm_provider": llm_provider,
        "model": model
    }

    try:
        response = requests.post(f"{BACKEND_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def check_backend_status() -> bool:
    """
    Check if the backend API is running.

    Returns:
        bool: True if backend is healthy, False otherwise.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_backend_logs(limit: int = 100) -> List[dict]:
    """
    Fetch backend logs from the API.

    Args:
        limit: Maximum number of logs to retrieve

    Returns:
        List[dict]: List of log entries with timestamp, level, and message
    """
    try:
        response = requests.get(f"{BACKEND_URL}/logs", params={"limit": limit})
        response.raise_for_status()
        return response.json().get("logs", [])
    except requests.exceptions.RequestException:
        return []


def clear_backend_logs() -> bool:
    """
    Clear all backend logs.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.post(f"{BACKEND_URL}/logs/clear")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_log_level_color(level: str) -> str:
    """
    Get color for log level.

    Args:
        level: Log level (INFO, WARNING, ERROR, DEBUG)

    Returns:
        str: Hex color code
    """
    colors = {
        "INFO": "#28a745",      # Green
        "WARNING": "#ffc107",   # Yellow
        "ERROR": "#dc3545",     # Red
        "DEBUG": "#6f42c1",     # Purple
        "CRITICAL": "#e83e8c"   # Pink
    }
    return colors.get(level.upper(), "#6c757d")  # Default gray


def format_log_message(log_entry: dict) -> str:
    """
    Format a log entry for display.

    Args:
        log_entry: Log entry dictionary with timestamp, level, message

    Returns:
        str: Formatted HTML string
    """
    timestamp = log_entry.get("timestamp", "")
    level = log_entry.get("level", "INFO")
    message = log_entry.get("message", "")

    color = get_log_level_color(level)

    return f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        border-left: 3px solid {color};
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 12px;
        animation: slideIn 0.3s ease-out;
    ">
        <div style="color: {color}; font-weight: bold; margin-bottom: 2px;">
            {level}
        </div>
        <div style="color: #495057; margin-bottom: 2px;">
            {timestamp}
        </div>
        <div style="color: #212529; word-wrap: break-word;">
            {message}
        </div>
    </div>
    """


def create_log_viewer():
    """
    Create a cool animated log viewer component.
    """
    st.markdown("""
    <style>
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .log-container {
        max-height: 400px;
        overflow-y: auto;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #dee2e6;
    }

    .log-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }

    .log-stats {
        display: flex;
        justify-content: space-around;
        margin: 10px 0;
        flex-wrap: wrap;
    }

    .stat-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 11px;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="log-header"><h3>🔥 Live Backend Logs</h3></div>', unsafe_allow_html=True)

    # Initialize session state for logs if not exists
    if 'current_logs' not in st.session_state:
        st.session_state.current_logs = []
    if 'last_log_count' not in st.session_state:
        st.session_state.last_log_count = 0

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Logs", key="refresh_logs"):
            # Force refresh logs
            st.session_state.current_logs = get_backend_logs(200)
            st.session_state.last_log_count = len(st.session_state.current_logs)
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Logs", key="clear_logs"):
            if clear_backend_logs():
                st.session_state.current_logs = []
                st.session_state.last_log_count = 0
                st.success("Logs cleared!")
                st.rerun()
            else:
                st.error("Failed to clear logs")

    with col3:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True, key="auto_refresh")

    # Create placeholders for dynamic content
    logs_placeholder = st.empty()
    stats_placeholder = st.empty()

    # Get fresh logs
    current_logs = get_backend_logs(200)

    # Check if logs have changed
    logs_changed = len(current_logs) != st.session_state.last_log_count

    # Update session state
    if logs_changed or not st.session_state.current_logs:
        st.session_state.current_logs = current_logs
        st.session_state.last_log_count = len(current_logs)

    # Use current logs from session state
    logs = st.session_state.current_logs

    if not logs:
        logs_placeholder.info("📭 No logs available. Backend may not be running or no activity yet.")
        return

    # Log statistics
    log_levels = {}
    for log in logs:
        level = log.get("level", "INFO")
        log_levels[level] = log_levels.get(level, 0) + 1

    # Display stats
    with stats_placeholder:
        st.markdown('<div class="log-stats">', unsafe_allow_html=True)
        for level, count in log_levels.items():
            color = get_log_level_color(level)
            st.markdown(f'<div class="stat-badge" style="color: {color}">{level}: {count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display logs
    with logs_placeholder:
        st.markdown('<div class="log-container">', unsafe_allow_html=True)

        # Show logs in reverse chronological order (newest first)
        for log_entry in reversed(logs[-50:]):  # Show last 50 logs
            st.markdown(format_log_message(log_entry), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Auto refresh mechanism - only update logs, not rerun whole page
    if auto_refresh:
        time.sleep(2)  # Wait 2 seconds
        # Only rerun if logs have actually changed to avoid unnecessary refreshes
        fresh_logs = get_backend_logs(200)
        if len(fresh_logs) != len(logs):
            st.session_state.current_logs = fresh_logs
            st.session_state.last_log_count = len(fresh_logs)
            st.rerun()


def main():
    # Header
    st.markdown('<h1 class="main-header">📚 DocuMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">Knowledge-Base Search Engine</p>', unsafe_allow_html=True)

    # Backend health check
    if not check_backend_status():
        st.error("⚠️ Backend API is not running. Please start the backend first.")
        st.stop()
    else:
        st.sidebar.success("✅ Backend API connected!")

    # Sidebar Log Viewer
    with st.sidebar:
        st.markdown("---")  # Separator
        create_log_viewer()

    # Session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = False
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = ""
    if 'is_summarization' not in st.session_state:
        st.session_state.is_summarization = False

    # Upload or Paste Text Section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Documents or Paste Text")

        # Upload files
        uploaded_files = st.file_uploader(
            "Choose documents (PDF, DOCX, TXT)",
            type=["pdf","docx","txt"],
            accept_multiple_files=True
        )

        # Text area for direct input
        pasted_text = st.text_area(
            "Or paste text here directly",
            height=150,
            placeholder="Paste any text here to query..."
        )

        if st.button("🚀 Process Input"):
            if uploaded_files or pasted_text.strip():
                with st.spinner("Processing input..."):
                    result = {"successful_uploads":0, "total_chunks":0}

                    # Process uploaded files
                    if uploaded_files:
                        upload_result = upload_files_to_backend(uploaded_files)
                        if "error" not in upload_result:
                            result["successful_uploads"] += upload_result.get("successful_uploads",0)
                            result["total_chunks"] += upload_result.get("total_chunks",0)
                        else:
                            st.error(f"File upload failed: {upload_result['error']}")

                    # Process pasted text
                    if pasted_text.strip():
                        text_result = upload_text_to_backend(pasted_text)
                        if "error" not in text_result:
                            result["successful_uploads"] += text_result.get("successful_uploads",0)
                            result["total_chunks"] += text_result.get("total_chunks",0)
                        else:
                            st.error(f"Pasted text processing failed: {text_result['error']}")

                    st.success(f"✅ Processed {result['successful_uploads']} document(s)/text input")
                    st.info(f"📊 Generated {result['total_chunks']} text chunks for search")
                    st.session_state.uploaded_files = True
            else:
                st.warning("Please upload a document or paste some text.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Query Section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Query Knowledge Base")
        query = st.text_input("Enter your question or ask for a summary")
        
        # Advanced options in an expander
        with st.expander("Advanced Options"):
            include_sources = st.checkbox("Include source citations", value=False)
            
            # LLM Provider selection
            llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "ollama", "huggingface"],
                index=0,
                help="Select the LLM provider to use for generating answers"
            )
            
            # Model selection based on provider
            if llm_provider == "openai":
                model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                    index=0
                )
            elif llm_provider == "ollama":
                model = st.selectbox(
                    "Ollama Model",
                    ["llama3", "llama3:8b", "mistral", "mistral:7b"],
                    index=0
                )
            else:  # huggingface
                model = st.selectbox(
                    "Hugging Face Model",
                    ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"],
                    index=0
                )
        
        search_col, summarize_col = st.columns(2)
        
        with search_col:
            if st.button("🔎 Search") and query:
                with st.spinner("Generating answer..."):
                    query_result = query_backend(query, include_sources, llm_provider, model)
                    if "error" not in query_result:
                        st.session_state.last_query = query
                        st.session_state.last_answer = query_result.get("answer","No answer found")
                        st.session_state.is_summarization = query_result.get("is_summarization", False)
                    else:
                        st.error(f"❌ Query failed: {query_result['error']}")
                        
        with summarize_col:
            if st.button("📝 Summarize Documents"):
                with st.spinner("Generating summary..."):
                    summarize_query = "Summarize the main points of these documents"
                    query_result = query_backend(summarize_query, include_sources, llm_provider, model)
                    if "error" not in query_result:
                        st.session_state.last_query = summarize_query
                        st.session_state.last_answer = query_result.get("answer","No answer found")
                        st.session_state.is_summarization = True
                    else:
                        st.error(f"❌ Summary failed: {query_result['error']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Results Section
    if st.session_state.last_answer:
        with st.container():
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            
            # Use different header based on whether it's a summarization or a query
            if hasattr(st.session_state, 'is_summarization') and st.session_state.is_summarization:
                st.subheader("📝 Document Summary")
                st.markdown("**Summary Request:**")
            else:
                st.subheader("📋 Answer")
                st.markdown("**Your Question:**")
                
            st.write(st.session_state.last_query)
            
            if hasattr(st.session_state, 'is_summarization') and st.session_state.is_summarization:
                st.markdown("**Summary:**")
            else:
                st.markdown("**Answer:**")
                
            st.write(st.session_state.last_answer)
            st.markdown('</div>', unsafe_allow_html=True)


# Example usage in comments (removed long example/features block to keep title at the top)

if __name__ == "__main__":
    main()