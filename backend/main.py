from fastapi import FastAPI, HTTPException
app = FastAPI()
from datetime import datetime
from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


@app.get("/v1/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/embeddings:batchUpsert")
async def embeddings():
    []
    return {"message": "Photo uploaded successfully"}

@app.post("/v1/search")
async def search_photos(query: str):
    []
    return {"results": []}


#Last ingest time
@app.get("/v1/stats/ingestion")
async def ingestion_stats():
    []
    return {"total_photos": 0, "total_users": 0}