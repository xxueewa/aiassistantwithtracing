from fastapi import FastAPI
from .routes import ingest
from .routes import query
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Register routes
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])