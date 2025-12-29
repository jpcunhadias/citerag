#!/bin/bash
# Start FastAPI backend server

echo "Starting FastAPI backend server..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

