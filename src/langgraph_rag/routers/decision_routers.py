"""LangGraphフロー用の条件分岐ロジック。"""

from __future__ import annotations

from ..models.schemas import QuestionCategory
from ..state import RAGState


def category_router(state: RAGState) -> str:
    """質問カテゴリに応じた分岐を返す。"""

    classification = state.get("classification")
    if classification is None or classification.category == QuestionCategory.OTHER:
        return "out_of_scope"
    return "retrieve_documents"


def relevance_router(state: RAGState) -> str:
    """関連ドキュメント有無で分岐を返す。"""

    if len(state.get("relevant_docs", [])) == 0:
        return "no_relevant_docs"
    return "generate_answer"
