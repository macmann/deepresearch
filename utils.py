"""Utility functions for document processing and vector storage."""

import os
import uuid
from typing import List

from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def load_file(path: str) -> List[Document]:
    """Load a single file and return a list of ``Document`` chunks.

    The loader is chosen based on the file extension. Each returned document will
    contain the original file name and file type in its metadata.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in {".docx", ".doc"}:
        loader = UnstructuredWordDocumentLoader(path)
    elif ext in {".xlsx", ".xls"}:
        loader = UnstructuredExcelLoader(path)
    elif ext in {".pptx", ".ppt"}:
        loader = UnstructuredPowerPointLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Attach metadata for tracing
    for chunk in chunks:
        chunk.metadata.update({"file_name": os.path.basename(path), "file_type": ext})

    return chunks


def embed_documents(docs: List[Document], embeddings: OpenAIEmbeddings) -> List[List[float]]:
    """Generate embedding vectors for the supplied documents."""
    texts = [d.page_content for d in docs]
    return embeddings.embed_documents(texts)


def upsert_vectors(
    client: QdrantClient,
    collection_name: str,
    docs: List[Document],
    embeddings: OpenAIEmbeddings,
) -> None:
    """Embed documents and upsert them into a Qdrant collection."""
    vectors = embed_documents(docs, embeddings)
    points = []
    for doc, vector in zip(docs, vectors):
        payload = doc.metadata or {}
        points.append(
            qmodels.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        )

    client.upsert(collection_name=collection_name, points=points)


def query_vectors(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    top_k: int = 5,
):
    """Return the most similar documents in Qdrant for the given query."""
    vector = embeddings.embed_query(query)
    return client.search(collection_name=collection_name, query_vector=vector, limit=top_k)
