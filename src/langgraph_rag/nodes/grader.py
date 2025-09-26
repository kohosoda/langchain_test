"""関連性評価ノードの実装。"""

from __future__ import annotations

from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models.schemas import RelevanceGrade
from ..state import RAGState


GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ドキュメントが質問に関連しているかを厳格に評価してください。
関連性が高い場合のみ'true'を返し、曖昧または無関係な場合は'false'を返してください。""",
        ),
        (
            "human",
            """質問: {question}

ドキュメント:
{document}

このドキュメントは質問に答えるために有用ですか？""",
        ),
    ]
)


def relevance_grader(state: RAGState) -> dict[str, Any]:
    """検索結果ドキュメントの関連性を評価する。"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(RelevanceGrade)
    chain = GRADER_PROMPT | structured_llm

    relevance_grades = []
    relevant_docs = []

    for doc in state.get("documents", []):
        grade = chain.invoke(
            {
                "question": state["question"],
                "document": doc.page_content,
            }
        )
        relevance_grades.append(grade)
        if grade.is_relevant:
            relevant_docs.append(doc)

    return {
        "relevance_grades": relevance_grades,
        "relevant_docs": relevant_docs,
    }
