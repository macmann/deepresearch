import os
import uuid
import json
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Location of stored reports
REPORTS_FILE = "reports.json"

# Initialize FastAPI application
app = FastAPI(title="DeepResearch API")

# Allow CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_reports() -> dict:
    """Load previously generated reports from disk."""
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If file exists but is invalid, ignore and start fresh
            return {}
    return {}


def save_reports(reports: dict) -> None:
    """Persist reports to disk."""
    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)


# In-memory store for demo purposes
REPORTS = load_reports()

# Qdrant client configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "research"

# Initialize embeddings and LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)

# Setup Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Ensure collection exists with appropriate vector size
try:
    qdrant_client.get_collection(QDRANT_COLLECTION)
except Exception:
    dim = len(embeddings.embed_query("test"))
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )


class ResearchRequest(BaseModel):
    prompt: str
    top_k: int = 5


@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload documents, split them and store embeddings in Qdrant."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    documents: List[Document] = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Choose appropriate loader based on extension
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext in {".docx", ".doc"}:
                loader = UnstructuredWordDocumentLoader(tmp_path)
            elif ext in {".xlsx", ".xls"}:
                loader = UnstructuredExcelLoader(tmp_path)
            elif ext in {".pptx", ".ppt"}:
                loader = UnstructuredPowerPointLoader(tmp_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

            documents.extend(loader.load())
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Split documents into chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Create vector store interface
    vectordb = Qdrant(client=qdrant_client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)

    # Add documents into the vector store
    vectordb.add_documents(docs)

    return {"message": f"Processed {len(files)} file(s) and stored {len(docs)} chunks."}


@app.post("/research")
async def research(request: ResearchRequest):
    """Retrieve relevant chunks and generate a research report."""
    vectordb = Qdrant(client=qdrant_client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)

    # Search for similar documents
    try:
        docs = vectordb.similarity_search(request.prompt, k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": "You are an AI research assistant. Use the provided context to answer the question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.prompt}"},
    ]

    try:
        response = llm.invoke(messages)
        report_text = response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    # Save the generated report
    report_id = str(uuid.uuid4())
    REPORTS[report_id] = {"id": report_id, "prompt": request.prompt, "report": report_text}
    save_reports(REPORTS)

    return {"id": report_id, "report": report_text}


@app.get("/reports")
async def list_reports():
    """List all previously generated reports."""
    return list(REPORTS.values())


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Fetch details of a single report."""
    report = REPORTS.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
