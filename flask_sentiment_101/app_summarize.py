from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# 요약 모델 (한국어)
summarizer = pipeline(
    "summarization",
    model="digit82/kobart-summarization",
    tokenizer="digit82/kobart-summarization",
    device=0  # GPU 없으면 -1
)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "")

        if input_text.strip():
            result = summarizer(
                input_text,
                max_length=128,
                min_length=30,
                do_sample=False
            )
            summary = result[0]["summary_text"]

    return render_template(
        "index2.html",
        summary=summary,
        input_text=input_text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
