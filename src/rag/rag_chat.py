# query_with_llm.py
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Tokenizerの並列処理警告を無効化
os.environ["TOKENIZERS_PARALLELISM"] = "false"


PERSIST_DIR = "../../chroma_db"
COLLECTION_NAME = "rag_docs"

load_dotenv()  # .env の OPENAI_API_KEY を読む

SYSTEM_PROMPT = """あなたは有能なリサーチアシスタントです。
与えられたコンテキスト（ドキュメント抜粋）のみを根拠に、質問に日本語で簡潔かつ正確に答えてください。
不確かな点は正直に「不明」と述べ、推測で補わないでください。
必要なら箇条書きを使って構いません。"""

PROMPT_TMPL = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "質問: {question}\n\n# コンテキスト:\n{context}\n\nコンテキストの情報に基づいて回答してください。"),
    ]
)


def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(parts)


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_TMPL
        | llm
        | StrOutputParser()
    )

    print("質問を入力してください（例: '要件定義の前提条件は？'）")
    q = input("> ").strip()
    result = rag_chain.invoke(q)
    print("\n--- 回答 ---")
    print(result)


if __name__ == "__main__":
    main()
