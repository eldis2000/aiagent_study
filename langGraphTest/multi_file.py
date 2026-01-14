from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

import pymysql


# =====================================================
# MySQL 설정
# =====================================================
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "tschoi",
    "password": "ts1004",
    "database": "spring_db",
    "charset": "utf8mb4"
}


# =====================================================
# State 정의
# =====================================================
class MyState(TypedDict):
    text: str
    sentiment: str


# =====================================================
# Ollama 설정
# =====================================================
llm = ChatOllama(
    model="qwen2:7b",   # llama3, mistral 등 가능
    temperature=0
)


# =====================================================
# 감정 분석 Node
# =====================================================
def analyze_sentiment(state: MyState) -> MyState:
    messages = [
        SystemMessage(
            content=(
                "너는 문장의 감정을 분류하는 AI다.\n"
                "반드시 아래 두 단어 중 하나만 출력하라.\n"
                "- positive\n"
                "- negative\n"
                "설명은 절대 하지 마라."
            )
        ),
        HumanMessage(content=state["text"])
    ]

    result = llm.invoke(messages).content.strip().lower()

    if result not in ("positive", "negative"):
        result = "negative"

    state["sentiment"] = result
    return state


# =====================================================
# 분기 Node
# =====================================================
def positive_node(state: MyState) -> MyState:
    print(f"✅ POSITIVE | {state['text']}")
    return state


def negative_node(state: MyState) -> MyState:
    print(f"❌ NEGATIVE | {state['text']}")
    return state


# =====================================================
# MySQL 저장 Node
# =====================================================
def save_to_mysql(state: MyState) -> MyState:
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()

    sql = """
        INSERT INTO sentiment_log (text, sentiment)
        VALUES (%s, %s)
    """
    cursor.execute(sql, (state["text"], state["sentiment"]))

    conn.commit()
    cursor.close()
    conn.close()

    print("💾 DB 저장 완료")
    return state


# =====================================================
# 라우팅
# =====================================================
def route_by_sentiment(state: MyState) -> str:
    return state["sentiment"]


# =====================================================
# LangGraph 구성
# =====================================================
graph = StateGraph(MyState)

graph.add_node("analyze", analyze_sentiment)
graph.add_node("positive", positive_node)
graph.add_node("negative", negative_node)
graph.add_node("save", save_to_mysql)

graph.set_entry_point("analyze")

graph.add_conditional_edges(
    "analyze",
    route_by_sentiment,
    {
        "positive": "positive",
        "negative": "negative"
    }
)

graph.add_edge("positive", END)
graph.add_edge("negative", "save")
graph.add_edge("save", END)

app = graph.compile()


# =====================================================
# 멀티 문서 처리
# =====================================================
if __name__ == "__main__":
    documents: List[str] = [
        "이 제품 정말 좋아요",
        "배송이 너무 느려서 불만입니다",
        "가격 대비 괜찮은 편이에요",
        "완전 별로네요 다시는 안 씁니다"
    ]

    print("📄 멀티 문서 처리 시작\n")

    for doc in documents:
        app.invoke(
            {
                "text": doc,
                "sentiment": ""
            }
        )

    print("\n✅ 모든 문서 처리 완료")
