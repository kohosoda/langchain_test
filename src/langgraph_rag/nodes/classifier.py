"""質問分類ノードの実装。"""

from __future__ import annotations

from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models.schemas import CategoryClassification
from ..state import RAGState


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """あなたは質問を以下のカテゴリに分類する専門家です：
- ai_dev: AI開発、機械学習、MLOps、データサイエンス関連
- product: ChatBot Pro製品仕様、機能、料金体系関連
- tech: プログラミング、システム設計、技術一般
- other: 上記以外（挨拶、雑談、無関係な質問など）

質問の内容を慎重に分析し、最も適切なカテゴリを選択してください。""",
        ),
        ("human", "質問: {question}"),
    ]
)


def question_classifier(state: RAGState) -> dict[str, Any]:
    """質問をカテゴリ分類する。"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(CategoryClassification)
    chain = CLASSIFIER_PROMPT | structured_llm

    classification = chain.invoke({"question": state["question"]})

    return {"classification": classification}
