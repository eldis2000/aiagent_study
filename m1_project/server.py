from flask import Flask, request, jsonify
from flask_cors import CORS 
from google import genai
import os 

# =========================================================
# 🚨🚨🚨 새로 발급받은 API 키를 여기에 입력하세요! 🚨🚨🚨
API_KEY = ""
# =========================================================

app = Flask(__name__)
# CORS 설정: 8000번 포트에서 오는 요청을 허용합니다.
CORS(app) 

# Gemini 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', 'Hello')

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt]
        )
        
        return jsonify({
            'success': True,
            'reply': response.text
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'success': False,
            'error': "API 호출 중 오류 발생. 키 또는 할당량을 확인하세요."
        }), 500

if __name__ == '__main__':
    # Flask 서버를 5000번 포트에서 실행
    app.run(port=5000, debug=True)