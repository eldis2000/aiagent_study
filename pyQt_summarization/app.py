import os
import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QRadioButton, QMessageBox
)
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------
# Config
# --------------------------------
UPLOAD_DIR = "uploads"
MODEL_NAME = "lcw99/t5-base-korean-text-summary"

device = "cuda" if torch.cuda.is_available() else "cpu"
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
# Summarization
# --------------------------------
def summarize_text(text: str, summary_type: str) -> str:
    if not text.strip():
        return "요약할 텍스트가 없습니다."

    if summary_type == "long":
        max_len = 400
        min_len = 150
        beams = 4
    else:
        max_len = 200
        min_len = 60
        beams = 2

    inputs = tokenizer(
        text[:3000],
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
# PyQt UI
# --------------------------------
class PdfSummaryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF 요약기 (transformers)")
        self.setGeometry(300, 200, 700, 600)

        self.pdf_path = None

        # Widgets
        self.label = QLabel("PDF 파일을 선택하세요")
        self.btn_select = QPushButton("📂 PDF 선택")
        self.btn_summary = QPushButton("📝 요약하기")

        self.radio_short = QRadioButton("짧게 요약")
        self.radio_long = QRadioButton("자세히 요약")
        self.radio_short.setChecked(True)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)

        # Layouts
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_short)
        radio_layout.addWidget(self.radio_long)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_select)
        layout.addLayout(radio_layout)
        layout.addWidget(self.btn_summary)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

        # Signals
        self.btn_select.clicked.connect(self.select_pdf)
        self.btn_summary.clicked.connect(self.run_summary)

    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "PDF 선택", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.pdf_path = file_path
            self.label.setText(f"선택된 파일: {os.path.basename(file_path)}")

    def run_summary(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "경고", "PDF 파일을 먼저 선택하세요.")
            return

        self.text_result.setText("요약 중입니다... 잠시만 기다려주세요 ⏳")

        try:
            text = extract_text_from_pdf(self.pdf_path)
            summary_type = "long" if self.radio_long.isChecked() else "short"
            summary = summarize_text(text, summary_type)
            self.text_result.setText(summary)
        except Exception as e:
            QMessageBox.critical(self, "오류", str(e))

# --------------------------------
# Main
# --------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PdfSummaryApp()
    window.show()
    sys.exit(app.exec_())
