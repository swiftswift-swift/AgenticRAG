# app.py

from fastapi import FastAPI
from pydantic import BaseModel

from agent.ragGraph import graph_rag

app = FastAPI()

graph = graph_rag()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):

    result = graph.invoke({
        "main_query": request.query,
        "current_query": request.query,
        "documents": [],
        "sources": [],
        "retry_count": 0
    })

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }
