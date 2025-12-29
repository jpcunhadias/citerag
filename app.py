"""Streamlit application for the RAG documentation search system."""

import logging
import sys

import streamlit as st

from src.search import generate_answer, retrieve_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect CLI mode
if len(sys.argv) > 1 and sys.argv[1] in ("ingest", "search", "ask"):
    # Route to CLI handler
    from src.cli import main

    sys.exit(main())

# Page configuration
st.set_page_config(
    page_title="Documentation Search (RAG)",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Documentation Search (RAG)")

# Sidebar
with st.sidebar:
    st.header("Filters")
    library_filter = st.selectbox(
        "Library",
        options=[None, "pandas", "numpy", "sklearn"],
        format_func=lambda x: "All" if x is None else x,
    )
    version_filter = st.text_input("Version (optional)", value="")

    st.header("Settings")
    show_raw_results = st.checkbox("Show raw search results", value=False)
    enable_rag = st.checkbox("Enable RAG answer", value=True)

# Main area
query = st.text_input("Enter your query:", placeholder="e.g., How do I merge two DataFrames?")

if st.button("Search", type="primary") and query:
    filters = {}
    if library_filter:
        filters["library"] = library_filter
    if version_filter:
        filters["version"] = version_filter

    with st.spinner("Searching..."):
        # Retrieve context
        results = retrieve_context(query, top_k=5, filters=filters if filters else None)

        if show_raw_results:
            st.subheader("Search Results")
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i} (Score: {result.score:.4f})"):
                    if result.header:
                        st.markdown(f"**Section:** {result.header}")
                    st.markdown(f"**Text:**\n{result.text}")
                    if result.url:
                        st.markdown(f"**URL:** {result.url}")
                    if result.file_path:
                        st.markdown(f"**File:** {result.file_path}")

        if enable_rag and results:
            st.subheader("RAG Answer")
            answer = generate_answer(query, results)
            st.markdown(answer)

            with st.expander("Sources"):
                for i, result in enumerate(results, 1):
                    st.markdown(f"{i}. {result.header or 'N/A'}")
                    if result.url:
                        st.markdown(f"   - {result.url}")
