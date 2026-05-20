FROM python:3.11-slim

# PortAudio is required by the sounddevice package
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install stsservice dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.0.0" \
    "python-dotenv>=1.0.0" \
    "openai>=1.0.0" \
    "langgraph-sdk>=0.1.0" \
    "langsmith>=0.1.0" \
    "numpy>=1.26.0" \
    "sounddevice>=0.4.0"

# Copy stsservice source
COPY stsservice/ ./stsservice/

WORKDIR /app/stsservice

EXPOSE 8001

# LANGGRAPH_URL overrides the hardcoded http://localhost:2024 in main.py.
# When running via docker-compose, set: LANGGRAPH_URL=http://agentstudio:2024
# Pass secrets via environment variables: OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]