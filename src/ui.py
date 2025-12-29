"""Streamlit UI for RAG documentation search system."""

import logging

import pandas as pd
import streamlit as st

from src.api_client import APIClient, APIClientError
from src.config import API_BASE_URL, RAG_REFUSAL_MESSAGE
from src.models import Citation

logger = logging.getLogger(__name__)


def init_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = APIClient(base_url=API_BASE_URL)


def build_citations_data(citations: list[Citation]) -> list[dict[str, str]]:
    """
    Build a structured data list from citations for display.

    Args:
        citations: List of Citation objects

    Returns:
        List of dictionaries with citation data
    """
    citations_data = []
    for citation in citations:
        header = (
            citation.header
            if citation.header
            else (citation.title if citation.title else "-")
        )
        citations_data.append(
            {
                "ID": citation.label,
                "Score": f"{citation.score:.4f}" if citation.score is not None else "-",
                "File": citation.source_path,
                "Header": header,
            }
        )
    return citations_data


@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_collections(api_client: APIClient) -> list[str]:
    """
    Fetch available collections from API.

    Args:
        api_client: APIClient instance

    Returns:
        List of collection names
    """
    try:
        collections = api_client.get_collections()
        return collections
    except APIClientError as e:
        logger.error(f"Error fetching collections: {e}")
        return ["pandas_docs"]  # Fallback


def render() -> None:
    """Main render function for Streamlit UI."""
    # Page configuration
    st.set_page_config(
        page_title="Chat with Docs",
        page_icon="💬",
        layout="wide",
    )

    # Initialize session state
    init_state()

    # Get API client from session state
    api_client = st.session_state["api_client"]

    # Sidebar - Settings
    with st.sidebar:
        st.title("⚙️ Settings")

        # Get available collections
        available_collections = get_collections(api_client)

        if not available_collections:
            st.warning("No collections found. Please ingest documents first.")
            collection_name = "pandas_docs"
        else:
            # Default to pandas_docs if available, otherwise first collection
            default_idx = 0
            if "pandas_docs" in available_collections:
                default_idx = available_collections.index("pandas_docs")

            collection_name = st.selectbox(
                "Collection Name",
                options=available_collections,
                index=default_idx,
            )

        top_k = st.slider(
            "Top-K (Initial Results)", min_value=10, max_value=100, value=25, step=5
        )
        top_n = st.slider(
            "Top-N (Reranked Results)", min_value=1, max_value=10, value=5, step=1
        )
        use_reranker = st.checkbox("Use Reranker", value=True)
        debug_mode = st.checkbox("Debug Mode", value=False)

        # Clear Chat Button
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
            # Clear response metadata
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith("response_")]
            for k in keys_to_remove:
                del st.session_state[k]
            st.rerun()

    # Main title
    st.title("💬 Chat with Docs")

    # Display chat history with sources
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Restore sources and debug for assistant messages
            if message["role"] == "assistant":
                response_key = f"response_{idx}"
                if response_key in st.session_state:
                    response_data = st.session_state[response_key]

                    # Show sources if available
                    if response_data["citations"] and message["content"] != RAG_REFUSAL_MESSAGE and not message["content"].startswith("❌"):
                        with st.expander("📚 Sources Used"):
                            citations_data = build_citations_data(response_data["citations"])
                            df = pd.DataFrame(citations_data)
                            st.dataframe(df, width='stretch')

                    # Show debug if available
                    if response_data["debug_mode"] and response_data["context_used"]:
                        with st.expander("🔍 Debug Context"):
                            st.code(response_data["context_used"], language=None)

    # Chat input
    query = st.chat_input("Ask a question about the documentation...")

    if query:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(query)

        # Append user message to state
        st.session_state.messages.append({"role": "user", "content": query})

        # Process query
        try:
            # Show spinner and call API
            with st.spinner("Thinking..."):
                response = api_client.ask(
                    query=query,
                    collection=collection_name,
                    top_k=top_k,
                    top_n=top_n,
                    rerank=use_reranker,
                    debug=debug_mode,
                )

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response.answer)

                # Append assistant message to state (just the answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.answer}
                )

                # Store response metadata in session state for persistence
                response_key = f"response_{len(st.session_state.messages) - 1}"
                st.session_state[response_key] = {
                    "citations": response.citations,
                    "context_used": response.context_used,
                    "debug_mode": debug_mode,
                }

                # Sources Widget
                if response.answer != RAG_REFUSAL_MESSAGE and response.citations:
                    with st.expander("📚 Sources Used"):
                        # Create DataFrame from citations
                        citations_data = build_citations_data(response.citations)
                        df = pd.DataFrame(citations_data)
                        st.dataframe(df, width="stretch")

                # Debug Widget
                if debug_mode and response.context_used:
                    with st.expander("🔍 Debug Context"):
                        st.code(response.context_used, language=None)

        except APIClientError as e:
            # Extract user-friendly message
            error_str = str(e)
            if "Failed to connect" in error_str or "connect to API" in error_str:
                user_msg = (
                    "❌ **Connection Error**\n\n"
                    f"Unable to connect to API backend at {API_BASE_URL}. "
                    "Please ensure the FastAPI server is running. "
                    "Start it with: `uvicorn api.main:app --reload`"
                )
            elif "Ollama service unavailable" in error_str:
                user_msg = (
                    "❌ **Service Error**\n\n"
                    "Ollama service is unavailable. Please ensure Ollama is running "
                    "and accessible at http://localhost:11434"
                )
            else:
                user_msg = f"❌ **API Error**: {error_str[:200]}..." if len(error_str) > 200 else f"❌ **API Error**: {error_str}"

            logger.error(f"API client error: {error_str}")

            with st.chat_message("assistant"):
                st.markdown(user_msg)

                # Show technical details only in debug mode
                if debug_mode:
                    with st.expander("🔍 Technical Details"):
                        st.code(error_str, language=None)
                        st.exception(e)

                # Append to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": user_msg}
                )

        except Exception as e:
            error_str = str(e)
            user_msg = f"❌ **An error occurred**: {error_str[:200]}..." if len(error_str) > 200 else f"❌ **An error occurred**: {error_str}"

            logger.error(f"Unexpected error: {error_str}", exc_info=True)

            with st.chat_message("assistant"):
                st.markdown(user_msg)

                # Show full technical details only in debug mode
                if debug_mode:
                    with st.expander("🔍 Technical Details"):
                        st.code(error_str, language=None)
                        st.exception(e)

                # Append to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": user_msg}
                )
