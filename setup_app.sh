#!/bin/bash

HOST="qdrant"
PORT="6333"

echo "⏳ Waiting for Qdrant to be ready at $HOST:$PORT..."

# using `netcat` to check if Qdrant is available.
while ! nc -z $HOST $PORT; do
  sleep 1
done

echo "✅ Qdrant is up, starting the application..."

# python3 ./innovation_resolution.py

streamlit run app.py --server.port=8501 --server.address=0.0.0.0

