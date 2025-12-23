import os
import base64
import json
import traceback

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI


# =================================================
# .env 로드
# =================================================
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 가 .env 에 설정되어 있지 않습니다.")


# =================================================
# OpenAI Client
# =================================================
client = OpenAI(api_key=API_KEY)


# =================================================
# Utils
# =================================================
def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# =================================================
# Streamlit UI
# =================================================
st.set_page_config(
    page_title="🚦 OpenAI Vision 교통표지판 분류",
    layout="centered"
)

st.title("🚦 OpenAI Vision-LLM (Responses API)")
st.caption("대한민국 도로교통 표지판 이미지 분류")

uploaded = st.file_uploader(
    "교통표지판 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    if st.button("🔍 표지판 분석"):
        with st.spinner("OpenAI Vision 분석 중..."):
            try:
                # ---------------------------------
                # 이미지 Data URL 생성 (⭐ 핵심)
                # ---------------------------------
                img_bytes = uploaded.getvalue()
                img_b64 = encode_image_bytes(img_bytes)
                img_data_url = f"data:image/jpeg;base64,{img_b64}"

                # ---------------------------------
                # OpenAI Responses API 호출
                # ---------------------------------
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "너는 대한민국 도로교통법 기준의 교통표지판 분류기다."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": """
이미지에서 가장 명확하게 보이는 대한민국 교통표지판 1개만 판단하라.
광고판, 간판, 신호등, 차량 표시는 교통표지판이 아니다.

아래 JSON 형식만 출력하라.
설명 문장, 주석, 추가 텍스트는 절대 출력하지 마라.

sign_type은 반드시 아래 5개 중 하나의 한국어 값만 사용하라.
- 규제표지
- 경고표지
- 지시표지
- 보조표지
- 미확인

sign_name은 대한민국 도로교통 표지판의 통용되는 한국어 명칭으로 작성하라.
description은 운전자가 취해야 할 행동을 한 문장으로 작성하라.
confidence는 0.0~1.0 사이의 숫자로 작성하라.

{
  "sign_type": "",
  "sign_name": "",
  "description": "",
  "confidence": 0.0
}
"""
                                },
                                {
                                    "type": "input_image",
                                    "image_url": img_data_url
                                }
                            ]
                        }
                    ],
                    max_output_tokens=300
                )

            except Exception:
                st.error("❌ OpenAI Responses API 호출 중 오류 발생")
                st.subheader("📛 전체 에러 로그")
                st.code(traceback.format_exc())
                st.stop()

        # ---------------------------------
        # 응답 처리
        # ---------------------------------
        raw_text = response.output_text

        st.subheader("🧠 OpenAI Vision 원문 응답")
        st.code(raw_text)

        st.subheader("✅ 파싱된 결과")
        try:
            result = json.loads(raw_text)
            st.json(result)

            if float(result.get("confidence", 0)) < 0.7:
                st.warning("⚠️ 신뢰도가 낮아 미확인 처리 또는 검수가 필요합니다.")
        except Exception:
            st.error("❌ JSON 파싱 실패 (모델이 JSON 외 텍스트를 출력했을 수 있음)")
