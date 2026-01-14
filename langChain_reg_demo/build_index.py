import os
from glob import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "./docs"
INDEX_DIR = "./faiss_index"
EMBED_MODEL = "nomic-embed-text"


def load_documents():
    files = (
        glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
        + glob(os.path.join(DATA_DIR, "**/*.md"), recursive=True)
    )

    if not files:
        raise RuntimeError("❌ data/ 폴더에 문서가 없습니다.")

    docs = []
    for f in files:
        docs.extend(TextLoader(f, encoding="utf-8").load())

    return docs


def build_index():
    print("📄 문서 로딩 중...")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    print(f"✂️  문서 분할 완료: {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("🧠 임베딩 생성 & FAISS 인덱스 생성 중...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if os.path.exists(INDEX_DIR):
        print("⚠️ 기존 인덱스 덮어쓰기")
    vectorstore.save_local(INDEX_DIR)

    print("✅ FAISS 인덱스 생성 완료")


if __name__ == "__main__":
    build_index()
