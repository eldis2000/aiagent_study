from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Sentiment analysis pipeline (English default)
# clf = pipeline("sentiment-analysis")

clf = pipeline(
    "text-classification",  # sentiment-analysis 대신 text-classification (이 모델은 이렇게)
    model="tabularisai/multilingual-sentiment-analysis",
    return_all_scores=False  # True로 하면 모든 점수 나옴
)




@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            pred = clf(text)[0]
            result = {
                "label": pred["label"],
                "score": round(float(pred["score"]) * 100, 2)
            }

    return render_template("index.html", text=text, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
