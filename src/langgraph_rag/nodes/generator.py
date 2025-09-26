"""回答生成および検証ノードの実装。"""

from __future__ import annotations

from typing import Any, List

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models.schemas import GenerationResult, GroundingCheck, QuestionCategory
from ..state import RAGState


SYSTEM_PROMPT = """あなたは有能なリサーチアシスタントです。
提供されたドキュメントの内容に厳密に基づき、日本語で簡潔かつ正確に回答してください。
推測や未確認情報は含めず、不明な場合は「分かりません」と答えてください。"""


GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """質問: {question}

# コンテキスト:
{context}

ドキュメントの内容に基づいて回答を生成してください。
回答とともに、どの情報に基づいて回答したかの推論も含めてください。""",
        ),
    ]
)


CHECKER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """回答が提供されたドキュメントに基づいているかを厳格にチェックしてください。
ドキュメントに明確に記載されていない情報や推測を含む場合は'false'を返してください。""",
        ),
        (
            "human",
            """ドキュメント:
{documents}

生成された回答: {generation}

この回答は上記のドキュメントの内容に完全に基づいていますか？""",
        ),
    ]
)


def _format_docs(docs: List) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{idx}] ({source})\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "(ドキュメントなし)"


def answer_generator(state: RAGState) -> dict[str, Any]:
    """関連ドキュメントをもとに回答を生成する。"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(GenerationResult)
    chain = GENERATOR_PROMPT | structured_llm

    documents = state.get("relevant_docs", [])
    context = _format_docs(documents)

    generation = chain.invoke(
        {"question": state["question"], "context": context})

    return {"generation": generation}


def hallucination_checker(state: RAGState) -> dict[str, Any]:
    """生成回答がドキュメントに基づいているかを検証する。"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(GroundingCheck)
    chain = CHECKER_PROMPT | structured_llm

    documents_text = "\n\n".join(
        doc.page_content for doc in state.get("relevant_docs", []))

    grounding_check = chain.invoke(
        {
            "documents": documents_text or "(ドキュメントなし)",
            "generation": state["generation"].answer,
        }
    )

    return {"grounding_check": grounding_check}


def finalizer(state: RAGState) -> dict[str, Any]:
    """最終回答を決定する。"""

    classification = state.get("classification")
    if classification and classification.category == QuestionCategory.OTHER:
        final_answer = "申し訳ありませんが、その質問にはお答えできません"
        return {"final_answer": final_answer}

    grounding_check = state.get("grounding_check")
    generation = state.get("generation")

    if grounding_check is None or not grounding_check.is_grounded:
        final_answer = "分かりません"
    else:
        final_answer = generation.answer if generation else "分かりません"

    return {"final_answer": final_answer}
