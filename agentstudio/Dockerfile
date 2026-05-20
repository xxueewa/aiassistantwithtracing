FROM python:3.11-slim

WORKDIR /app

# Install ragcorelib (local dependency of agentstudio)
COPY ragcorelib/ ./ragcorelib/
RUN pip install --no-cache-dir ./ragcorelib/

# Copy agentstudio source (excludes .env and caches via .dockerignore)
COPY agentstudio/ ./agentstudio/

WORKDIR /app/agentstudio

# Install pinned deps from requirements.txt, skipping the editable -e . line
RUN grep -v "^-e " requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt

# Install the package itself
RUN pip install --no-cache-dir --no-deps .

EXPOSE 2024

# langgraph dev uses in-memory storage, suitable for this demo
# Pass secrets via environment variables: OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser"]