#!/bin/bash
# Start Streamlit UI (assumes backend is running)

echo "Starting Streamlit UI..."
echo "Note: Ensure FastAPI backend is running (use scripts/start_backend.sh)"
streamlit run ui.py

