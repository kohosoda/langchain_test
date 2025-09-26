from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
import os
from pathlib import Path

# Tokenizerの並列処理警告を無効化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOCS_DIR = "../docs"
PERSIST_DIR = "../chroma_db"


def load_documents(docs_dir: str):
    # 拡張子ごとにローダーを割り当て（最小構成）
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"{docs_dir} が見つかりません")

    documents = []
    for p in docs_path.rglob("*"):
        # 一旦 .txt と .md のみを対象とする
        if p.suffix.lower() in [".txt", ".md"]:
            documents.extend(TextLoader(
                str(p), autodetect_encoding=True).load())
        else:
            raise ValueError(f"未対応のファイル形式です: {p.name}（拡張子: {p.suffix}）")

    return documents


def split_documents(documents):
    # 日本語でも扱いやすいよう、文字数ベースで分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   # 文章量に応じて調整
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    return splitter.split_documents(documents)


def main():
    print("Loading documents...")
    documents = load_documents(DOCS_DIR)
    print(f"Loaded {len(documents)} docs")

    print("Splitting documents...")
    splits = split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Building embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating / updating Chroma index...")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="rag_docs",
    )
    print(f"Persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
