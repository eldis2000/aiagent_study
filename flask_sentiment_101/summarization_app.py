import os
import torch
from flask import Flask, render_template, request
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------
# Config
# ----------------------
UPLOAD_DIR = "uploads"
MODEL_NAME = "psyche/KoT5-base-korean-summarization"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------
# Load Model (1회)
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# ----------------------
# PDF → Text
# ----------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text.strip())
    return "\n".join(texts)

# ----------------------
# Summarization
# ----------------------
def summarize_text(text):
    inputs = tokenizer(
        text[:3000],
        return_tensors="pt",
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=256,
        min_length=80,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ----------------------
# Routes
# ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        file = request.files.get("pdf")
        if file and file.filename.endswith(".pdf"):
            path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(path)

            text = extract_text_from_pdf(path)
            summary = summarize_text(text)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
