import json
import re
import requests
import streamlit as st
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava:13b"

PROMPT = """
너는 교통표지판 분류기다.
이미지에서 가장 명확한 교통표지판 1개만 판단하라.
아래 JSON만 출력하라. 설명 문장이나 주석은 절대 출력하지 마라.

sign_type은 반드시 아래 5개 중 하나의 한국어 값만 선택하라.
- 규제표지
- 경고표지
- 지시표지
- 보조표지
- 미확인

다른 단어, 영문, 복수 값, 기호(|,/ 등)를 절대 사용하지 마라.

{
  "sign_type": "",
  "sign_name": "",
  "description": "",
  "confidence": 0.0
}
""".strip()

def extract_json(text: str):
    """
    LLaVA가 JSON 앞뒤로 말을 붙여도 JSON만 뽑아내기.
    """
    # 가장 바깥 {} 덩어리 추출
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_llava(image_bytes: bytes):
    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg")
    }
    # Ollama generate는 이미지 base64도 받지만, python에선 /api/generate + "images"가 정석
    # 여기서는 base64로 전송
    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")

st.set_page_config(page_title="🚦 LLaVA 교통표지판 분류기", layout="centered")
st.title("🚦 로컬 LLaVA(Ollama) 교통표지판 분류")

up = st.file_uploader("표지판 이미지 업로드", type=["jpg", "jpeg", "png"])

if up:
    img = Image.open(up)
    st.image(img, caption="업로드 이미지", use_container_width=True)

    if st.button("🔍 분류하기"):
        with st.spinner("LLaVA 분석 중..."):
            img_bytes = up.getvalue()
            out = call_llava(img_bytes)

        st.subheader("원문 출력")
        st.code(out)

        parsed = extract_json(out)
        if parsed:
            st.subheader("✅ JSON 결과")
            st.json(parsed)
        else:
            st.error("❌ JSON 파싱 실패 (모델이 JSON 외 텍스트를 섞어서 출력함)")
            st.info("해결: 프롬프트 강화 또는 extract_json 로직 보강")
