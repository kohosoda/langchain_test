"""LangGraph RAGワークフローで共有する状態定義。"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document

from .models.schemas import (
    CategoryClassification,
    GenerationResult,
    GroundingCheck,
    QuestionCategory,
    RelevanceGrade,
)


class RAGState(TypedDict, total=False):
    question: str
    classification: CategoryClassification
    documents: List[Document]
    relevance_grades: List[RelevanceGrade]
    relevant_docs: List[Document]
    generation: GenerationResult
    grounding_check: GroundingCheck
    final_answer: str
