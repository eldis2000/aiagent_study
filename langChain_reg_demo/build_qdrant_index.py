import os
from glob import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


DATA_DIR = "./docs"
COLLECTION_NAME = "doc_knowledge_base"
QDRANT_URL = "http://localhost:6333"
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

    print(f"✂️ 문서 분할 완료: {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    # ✅ 임베딩 차원 자동 계산
    dim = len(embeddings.embed_query("dimension check"))

    # ✅ 컬렉션 명시적 재생성
    if client.collection_exists(COLLECTION_NAME):
        print("⚠️ 기존 컬렉션 삭제")
        client.delete_collection(COLLECTION_NAME)

    print("🧠 Qdrant 컬렉션 생성 중...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        )
    )

    # ✅ 최신 LangChain-Qdrant 방식
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    print("⬆️ 문서 업서트 중...")
    vectorstore.add_documents(chunks)

    print("✅ Qdrant 인덱싱 완료")


if __name__ == "__main__":
    build_index()
