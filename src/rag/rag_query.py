# query.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# Tokenizerの並列処理警告を無効化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PERSIST_DIR = "../../chroma_db"
COLLECTION_NAME = "rag_docs"


def simple_search(query: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(query)
    return docs


def main():
    print("質問を入力してください（例: 'このプロジェクトの要件は？'）")
    query = input("> ").strip()
    docs = simple_search(query, k=4)

    print("\n--- 検索ヒット（上位）---")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        print(f"[{i}] source: {src}")
        print(d.page_content.replace("\n", " "))
        print("----")


if __name__ == "__main__":
    main()
