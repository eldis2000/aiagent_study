from typing import TypedDict
from langgraph.graph import StateGraph, END

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# =====================================================
# 1. State 정의
# =====================================================
class MyState(TypedDict):
    text: str
    sentiment: str


# =====================================================
# 2. Ollama LLM 설정
# =====================================================
llm = ChatOllama(
    model="qwen2:7b",   # llama3, mistral 등으로 변경 가능
    temperature=0
)


# =====================================================
# 3. 감정 분석 Node (LLM 기반 조건 판단)
# =====================================================
def analyze_sentiment(state: MyState) -> MyState:
    text = state["text"]

    messages = [
        SystemMessage(
            content=(
                "너는 문장의 감정을 분류하는 AI다.\n"
                "반드시 아래 두 단어 중 하나만 출력하라.\n"
                "- positive\n"
                "- negative\n"
                "설명이나 다른 문장은 절대 출력하지 마라."
            )
        ),
        HumanMessage(content=text)
    ]

    response = llm.invoke(messages).content.strip().lower()

    # 예외 처리 (환각 방지)
    if response not in ("positive", "negative"):
        response = "negative"

    state["sentiment"] = response
    return state


# =====================================================
# 4. 분기 Node
# =====================================================
def positive_node(state: MyState) -> MyState:
    print("✅ 긍정 문장입니다")
    return state


def negative_node(state: MyState) -> MyState:
    print("❌ 부정 문장입니다")
    return state


# =====================================================
# 5. 조건 라우터
# =====================================================
def route_by_sentiment(state: MyState) -> str:
    return state["sentiment"]


# =====================================================
# 6. LangGraph 구성
# =====================================================
graph = StateGraph(MyState)

graph.add_node("analyze", analyze_sentiment)
graph.add_node("positive", positive_node)
graph.add_node("negative", negative_node)

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
graph.add_edge("negative", END)

app = graph.compile()


# =====================================================
# 7. 실행부 (사용자 입력)
# =====================================================
if __name__ == "__main__":
    print("📢 문장을 입력하세요 (종료: Ctrl+C)\n")

    try:
        while True:
            user_text = input("> ").strip()
            if not user_text:
                continue

            result = app.invoke(
                {
                    "text": user_text,
                    "sentiment": ""
                }
            )

            print("📦 최종 State:", result)
            print("-" * 40)

    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
