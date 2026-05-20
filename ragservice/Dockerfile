FROM python:3.11-slim

WORKDIR /app

# Install ragcorelib (local dependency declared in pyproject.toml via uv.sources)
COPY ragcorelib/ ./ragcorelib/
RUN pip install --no-cache-dir ./ragcorelib/

# Install ragservice dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.0.0" \
    "langchain-text-splitters>=0.3.0" \
    "langsmith>=0.1.0"

# Copy ragservice source
COPY ragservice/ ./ragservice/

WORKDIR /app/ragservice

EXPOSE 8000

# Pass secrets via environment variables: OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]