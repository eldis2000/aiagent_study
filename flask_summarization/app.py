import os
import torch
from flask import Flask, render_template, request
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------
# Config
# --------------------------------
UPLOAD_DIR = "uploads"
MODEL_NAME = "lcw99/t5-base-korean-text-summary"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------
# Load model (ONCE)
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# --------------------------------
# PDF → Text
# --------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t.strip())
    return "\n".join(texts)

# --------------------------------
# Text Summarization
# --------------------------------
def summarize_text(text: str, summary_type: str = "short") -> str:
    if not text.strip():
        return "요약할 텍스트가 없습니다."

    # 요약 길이 옵션
    if summary_type == "long":
        max_len = 400
        min_len = 150
        beams = 4
    else:  # short
        max_len = 200
        min_len = 60
        beams = 2

    inputs = tokenizer(
        text[:3000],  # 길이 제한
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=beams,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --------------------------------
# Routes
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        file = request.files.get("pdf")
        summary_type = request.form.get("summary_type", "short")

        if file and file.filename.endswith(".pdf"):
            save_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(save_path)

            text = extract_text_from_pdf(save_path)
            summary = summarize_text(text, summary_type)

    return render_template("index.html", summary=summary)

# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
