# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from PIL import Image
from io import BytesIO
import base64
import os

# --- 0. 라이브러리 설치 필요 ---
# pip install flask flask-cors google-genai pillow

# --- 1. API 키 설정 (Gemini API Key를 환경 변수에서 가져오거나 여기에 직접 입력) ---
# 환경 변수 사용을 강력히 권장합니다.
# API_KEY = os.environ.get("GEMINI_API_KEY") 
API_KEY = ""  # <-- 여기에 실제 키를 넣어주세요.

if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
    print("❌ 경고: API 키가 설정되지 않았습니다. API_KEY 변수에 유효한 키를 입력하거나 환경 변수를 설정하세요.")

try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Gemini 클라이언트 초기화 오류: {e}")
    # 키가 잘못되었더라도 일단 서버는 시작되도록 합니다.
    pass


# --- 2. Flask 및 CORS 설정 ---
app = Flask(__name__)
# 클라이언트(브라우저)와 서버 포트가 다르므로 CORS 허용이 필요합니다.
CORS(app) 

# --- 3. Base64 디코딩 함수 ---
def decode_base64_image(base64_string):
    """Base64 Data URL을 PIL Image 객체로 디코딩 및 변환"""
    # 'data:image/png;base64,'와 같은 Data URL 헤더를 제거합니다.
    if ',' in base64_string:
        header, encoded = base64_string.split(',', 1)
    else:
        encoded = base64_string
    
    # Base64 문자열을 바이트 데이터로 디코딩
    binary_data = base64.b64decode(encoded)
    
    # 바이트 데이터를 메모리 스트림으로 읽어 PIL Image 객체 생성
    image = Image.open(BytesIO(binary_data))
    return image

# --- 4. 엔드포인트: /chat (멀티모달 처리) ---
@app.route('/chat', methods=['POST']) 
def multimodal_chat_local():
    """클라이언트로부터 Base64 이미지 데이터와 프롬프트를 받아 Gemini API로 전달합니다."""
    try:
        if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
             return jsonify({"success": False, "error": "Gemini API 키가 설정되지 않아 요청을 처리할 수 없습니다."}), 503

        data = request.get_json()
        base64_image = data.get('base64_image')
        prompt = data.get('prompt')

        if not base64_image or not prompt:
            return jsonify({"success": False, "error": "이미지 데이터 또는 질문(프롬프트)이 누락되었습니다."}), 400

        # 1. Base64 이미지 디코딩 및 PIL 객체로 변환
        image = decode_base64_image(base64_image)

        # 2. 멀티모달 컨텐츠 구성: [프롬프트 텍스트, 이미지 객체]
        content = [prompt, image]
        
        # 3. Gemini API 호출 (수정된 호출 방식 적용)
        response = client.models.generate_content(
            model='gemini-2.5-flash',              # 사용할 모델 이름 지정
            contents=content                       # 이미지와 텍스트 데이터 전달
        )
        
        # 4. 응답 반환
        return jsonify({
            "success": True,
            "reply": response.text 
        })

    except ValueError as ve:
        # decode_base64_image 함수에서 발생한 오류 (예: 잘못된 Base64 형식) 처리
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        print(f"❌ Flask 서버 처리 중 오류 발생: {e}")
        # API 오류 (예: 키 만료, 네트워크 문제 등) 포함
        return jsonify({"success": False, "error": f"서버 또는 API 처리 오류: {str(e)}"}), 500

# --- 5. 서버 실행 ---
if __name__ == '__main__':
    # 서버를 클라이언트가 요청하는 포트(5000)로 실행
    print("✨ Flask 서버가 http://127.0.0.1:5000 에서 실행 중입니다. ✨")
    app.run(debug=True, port=5000)