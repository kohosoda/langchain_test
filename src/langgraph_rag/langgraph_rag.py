"""LangGraphを用いたRAGチャットフローの定義。"""

from __future__ import annotations

from typing import Callable

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .models.schemas import QuestionCategory
from .nodes.classifier import question_classifier
from .nodes.generator import answer_generator, finalizer, hallucination_checker
from .nodes.grader import relevance_grader
from .nodes.retriever import document_retriever
from .routers.decision_routers import (
    category_router,
    relevance_router,
)
from .state import RAGState


load_dotenv()


def build_rag_workflow() -> Callable[[str], str]:
    """LangGraphベースのRAGワークフローを構築して実行可能オブジェクトを返す。"""

    graph = StateGraph(RAGState)

    # ノードを定義
    graph.add_node("classify", question_classifier)
    graph.add_node("retrieve", document_retriever)
    graph.add_node("grade", relevance_grader)
    graph.add_node("generate", answer_generator)
    graph.add_node("check", hallucination_checker)
    graph.add_node("finalize", finalizer)

    # グラフを構築
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", category_router, {
        "out_of_scope": "finalize",
        "retrieve_documents": "retrieve",
    })
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", relevance_router, {
        "no_relevant_docs": "finalize",
        "generate_answer": "generate",
    })
    graph.add_edge("generate", "check")
    graph.add_edge("check", "finalize")
    graph.add_edge("finalize", END)

    # コンパイル
    app = graph.compile()

    def runner(question: str) -> str:
        state: RAGState = {"question": question}

        final_result = None
        # 検証のために各ステップごとの結果を見れるようにする
        for step_output in app.stream(state):
            for node_name, node_result in step_output.items():
                print(f"[{node_name}] 実行完了")
                print(f"  State: {node_result}\n")
                final_result = node_result

        return final_result.get("final_answer", "分かりません") if final_result else "分かりません"

    return runner


def main() -> None:
    """対話的に質問を受け付ける。"""

    runner = build_rag_workflow()
    print("質問を入力してください。終了するにはCtrl+Cを押してください。")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                print("空の入力です。質問を入力してください。")
                continue
            answer = runner(question)
            print("\n--- 回答 ---")
            print(answer)
            print()
    except KeyboardInterrupt:
        print("\n終了します。")


if __name__ == "__main__":
    main()
