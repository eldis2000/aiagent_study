from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from qdrant_client import QdrantClient

COLLECTION_NAME = "doc_knowledge_base"
QDRANT_URL = "http://localhost:6333"

LLM_MODEL = "llama3"          # ollama list 결과와 일치
EMBED_MODEL = "nomic-embed-text"


def main():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 문서 기반 Q&A 챗봇이다. "
         "반드시 제공된 문서 컨텍스트만 근거로 답해라. "
         "문서에 없는 내용은 '데이터가 없습니다.'라고 답해라."),
        ("human",
         "질문: {question}\n\n"
         "문서 컨텍스트:\n{context}\n\n"
         "답변:")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    print("🤖 Qdrant 기반 문서 챗봇 실행 (exit / quit 종료)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        resp = chain.invoke(q)
        print(f"Bot: {resp.content}")


if __name__ == "__main__":
    main()
