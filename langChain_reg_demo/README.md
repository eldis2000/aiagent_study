## LangChain 을 이용한 RAG & Chatbot 

설치
- pip install -r requirements.txt

docs 폴더
- 학습할 자료(텍스트 파일)를 저장함

faiss_index
- 벡터DB

build_index.py
- 학습자료를 벡터 DB에 저장합니다
- python build_index.py

chat.py
- 챗봇, 학습한 내용을 바탕으로 질의에 답변 합니다.
- python chat.py 