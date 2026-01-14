import os

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

INDEX_DIR = "./faiss_index"
LLM_MODEL = "qwen2:7b"
EMBED_MODEL = "nomic-embed-text"


def main():
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError("❌ FAISS 인덱스가 없습니다. build_index.py를 먼저 실행하세요.")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
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

    print("🤖 문서 기반 챗봇 실행 (exit / quit 종료)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        resp = chain.invoke(q)
        print(f"Bot: {resp.content}")


if __name__ == "__main__":
    main()
