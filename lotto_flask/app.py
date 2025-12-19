from flask import Flask, render_template, request, jsonify
import random, ollama

app = Flask(__name__)

# --- 로또 번호 생성 ---
def generate_lotto_set():
    return sorted(random.sample(range(1, 46), 6))

@app.route("/")
def index():
    count = int(request.args.get("count", 1))
    results = [generate_lotto_set() for _ in range(count)]
    return render_template("index.html", results=results, count=count)

# --- Ollama 채팅 API ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "메시지가 없습니다."}), 400

    try:
        response = ollama.chat(
            model="qwen2:7b",  # 로컬에 설치된 모델명 (예: llama3, mistral 등)
            messages=[{"role": "user", "content": user_message}],
            options={"temperature": 0.7, "num_predict": 256},
        )
        reply = response["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
