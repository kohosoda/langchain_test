"""ベクトル検索ノードの実装。"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..state import RAGState


PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"


def _build_retriever():
    """Chromaベースのリトリーバーを生成する。"""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectordb.as_retriever(search_kwargs={"k": 5})


def document_retriever(state: RAGState) -> dict[str, Any]:
    """質問に関連するドキュメントを検索する。"""

    retriever = _build_retriever()
    documents = retriever.invoke(state["question"])
    return {"documents": documents}
