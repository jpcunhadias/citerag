"""Streamlit UI for RAG documentation search system."""

import logging

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT, RAG_REFUSAL_MESSAGE
from src.ingest import VectorService
from src.llm import OllamaClient, OllamaConnectionError
from src.rag import RAGService
from src.rerank import RerankerService
from src.search import SearchService

logger = logging.getLogger(__name__)


@st.cache_resource
def get_services() -> tuple[SearchService, RerankerService, OllamaClient]:
    """
    Initialize and cache heavy singleton services.

    Returns:
        Tuple of (SearchService, RerankerService, OllamaClient)
    """
    logger.info("Initializing services (cached)")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)
    reranker_service = RerankerService()
    llm_client = OllamaClient()

    return search_service, reranker_service, llm_client


def init_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_collections() -> list[str]:
    """Fetch available collections from Qdrant."""
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = qdrant_client.get_collections().collections
        return [col.name for col in collections]
    except Exception as e:
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

    # Sidebar - Settings
    with st.sidebar:
        st.title("⚙️ Settings")

        # Get available collections
        available_collections = get_collections()

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
                            citations_data = []
                            for citation in response_data["citations"]:
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
            # Retrieve cached services
            search_service, reranker_service, llm_client = get_services()

            # Create RAG service (lightweight, not cached)
            rag_service = RAGService(
                search_service=search_service,
                reranker_service=reranker_service,
                llm_client=llm_client,
            )

            # Show spinner and call RAG
            with st.spinner("Thinking..."):
                response = rag_service.ask(
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
                        citations_data = []
                        for citation in response.citations:
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

                        df = pd.DataFrame(citations_data)
                        st.dataframe(df, use_container_width=True)

                # Debug Widget
                if debug_mode and response.context_used:
                    with st.expander("🔍 Debug Context"):
                        st.code(response.context_used, language=None)

        except OllamaConnectionError as e:
            # Extract user-friendly message
            error_str = str(e)
            if "Failed to connect" in error_str:
                user_msg = "❌ **Connection Error**\n\nUnable to connect to Ollama. Please ensure Ollama is running and accessible at http://localhost:11434"
            else:
                user_msg = f"❌ **Error**: {error_str}"

            logger.error(f"Ollama connection error: {error_str}")

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

