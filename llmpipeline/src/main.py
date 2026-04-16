from fastapi import FastAPI
from langchainapp.routes import ingest
from langchainapp.routes import query

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Register routes
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])