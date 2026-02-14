#!/bin/bash

cleanup() {
    echo "Stopping services..."
    if [[ -n "$UVICORN_PID" ]]; then
        kill -TERM "$UVICORN_PID" 2>/dev/null
        wait "$UVICORN_PID" 2>/dev/null
    fi
    if [[ -n "$STREAMLIT_PID" ]]; then
        kill -TERM "$STREAMLIT_PID" 2>/dev/null
        wait "$STREAMLIT_PID" 2>/dev/null
    fi
}

trap cleanup SIGINT SIGTERM

# Lancer Uvicorn en arrière-plan
uvicorn app:app --host 0.0.0.0 --port 5000 &
UVICORN_PID=$!
echo "Uvicorn started with PID $UVICORN_PID"

# Lancer Streamlit en arrière-plan
streamlit run front.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!
echo "Streamlit started with PID $STREAMLIT_PID"

# Attendre les deux processus
wait $UVICORN_PID $STREAMLIT_PID
