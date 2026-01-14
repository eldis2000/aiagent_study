#               ┌──────────────┐
#               │    State     │
#               │  messages[]  │
#               └──────┬───────┘
#                      │
#           ┌──────────▼──────────┐
# 	        │  user_input Node    │
#           └──────────┬──────────┘
#                      │
#                   (조건 판단)
#        	┌──────────┴──────────┐
#           │                     │
#   ┌───────▼────────┐   ┌────────▼────────┐
#   │ search_node    │   │  llm_response   │
#   └───────┬────────┘   └────────┬────────┘
#           │                     │
# 	      [END]                 [END]

from typing import TypedDict
from langgraph.graph import StateGraph, END

class MyState(TypedDict):
	text: str
	sentiment: str
	

def analyze_sentiment(state: MyState) -> MyState:
    text = state["text"]
	
    if "좋아" in text:
        state["sentiment"] = "positive"
    else:
        state["sentiment"] = "negative"

    return state


def positive_node(state: MyState) -> MyState:
	print("긍정 문장입니다")
	return state

def negative_node(state: MyState) -> MyState:
	print("부정 문장입니다")
	return state

def route_by_sentiment(state: MyState) -> str:
	if state["sentiment"] == "positive":
		return "positive"
	else:
		return "negative"
	
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

result = app.invoke(
	{
		"text": "이 제품 정말 좋아요",
		"sentiment": ""
	}
)

print("최종 State:", result)