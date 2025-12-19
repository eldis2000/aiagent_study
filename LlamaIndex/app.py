import os
import json
import random
import streamlit as st

from pypdf import PdfReader

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "📘 PDF 기반 OX 퀴즈 생성기 (LlamaIndex + Qwen2)"
LLM_MODEL = "qwen2:7b"
EMBED_MODEL = "nomic-embed-text"  # Ollama embedding 모델(가볍고 많이 씀)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


# -----------------------------
# Utils
# -----------------------------
def read_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(t)
    return "\n\n".join(texts)


# def build_index_from_text(text: str) -> VectorStoreIndex:
#     # LlamaIndex 글로벌 설정
#     Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120)
#     Settings.embed_model = OllamaEmbedding(model=EMBED_MODEL)
#     Settings.chunk_size = CHUNK_SIZE
#     Settings.chunk_overlap = CHUNK_OVERLAP

#     doc = Document(text=text, metadata={"source": "uploaded_pdf"})
#     index = VectorStoreIndex.from_documents([doc])
#     return index
def build_index_from_text(text: str) -> VectorStoreIndex:
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=120,
        base_url="http://localhost:11434"
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434"
    )

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    doc = Document(text=text, metadata={"source": "uploaded_pdf"})
    index = VectorStoreIndex.from_documents([doc])
    return index



def generate_ox_questions(query_engine, n_questions: int = 10) -> list[dict]:
    """
    반환 형식:
    [
      {"q": "...", "answer": "O" or "X", "explain": "...", "evidence": "..."},
      ...
    ]
    """
    prompt = f"""
너는 O/X 퀴즈 출제자다.
주어진 문서 내용만 근거로 O/X 문제 {n_questions}개를 만들어라.
- 답은 반드시 "O" 또는 "X"로만.
- 각 문항은 한 문장으로 명확하게.
- 각 문항마다 근거가 되는 문장(또는 핵심 구절) 1개를 evidence로 포함.
- 설명(explain)은 1~2문장.

아래 JSON 배열 형식으로만 출력해라. (코드블록 금지)
[
  {{"q":"문제","answer":"O","explain":"해설","evidence":"근거문장"}},
  ...
]
""".strip()

    # 문서 기반으로 생성하도록 query_engine에 질의
    resp = query_engine.query(prompt)
    raw = str(resp).strip()

    # LLM이 JSON 앞뒤에 텍스트를 붙이는 경우 대비: 가장 바깥 []만 잘라 파싱 시도
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
        # 간단 검증
        cleaned = []
        for item in data:
            q = str(item.get("q", "")).strip()
            a = str(item.get("answer", "")).strip().upper()
            explain = str(item.get("explain", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            if q and a in ("O", "X"):
                cleaned.append({"q": q, "answer": a, "explain": explain, "evidence": evidence})
        return cleaned
    except Exception:
        return []


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("⚙️ 설정")
    num_q = st.slider("문항 수", 5, 30, 10, 1)
    shuffle_q = st.checkbox("문항 섞기", value=True)
    st.caption("LLM: qwen2:7b (Ollama), Embedding: nomic-embed-text")

uploaded = st.file_uploader("PDF 업로드", type=["pdf"])

if "index" not in st.session_state:
    st.session_state.index = None
if "questions" not in st.session_state:
    st.session_state.questions = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False


if uploaded:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1) PDF 읽기 / 인덱싱")
        if st.button("📌 인덱스 생성", type="primary"):
            with st.spinner("PDF 텍스트 추출 중..."):
                text = read_pdf_text(uploaded)

            if not text.strip():
                st.error("PDF에서 텍스트를 추출하지 못했습니다. (스캔 PDF면 OCR이 필요할 수 있어요)")
            else:
                with st.spinner("LlamaIndex 인덱스 생성 중..."):
                    st.session_state.index = build_index_from_text(text)
                st.success("인덱스 생성 완료!")

    with col2:
        st.subheader("2) OX 퀴즈 생성")
        can_make = st.session_state.index is not None
        if st.button("🧠 퀴즈 생성", disabled=not can_make):
            with st.spinner("문서 기반 OX 퀴즈 생성 중..."):
                qe = st.session_state.index.as_query_engine(similarity_top_k=4)
                qs = generate_ox_questions(qe, n_questions=num_q)

            if not qs:
                st.error("퀴즈 생성에 실패했어요. (모델 출력이 JSON이 아니거나 근거 부족)")
            else:
                if shuffle_q:
                    random.shuffle(qs)
                st.session_state.questions = qs
                st.session_state.submitted = False
                st.session_state.score = 0
                st.success(f"퀴즈 {len(qs)}개 생성 완료!")

st.divider()

if st.session_state.questions:
    st.subheader("📝 퀴즈 풀기")

    answers = []
    for i, item in enumerate(st.session_state.questions, start=1):
        with st.container(border=True):
            st.markdown(f"**Q{i}.** {item['q']}")
            choice = st.radio(
                "정답 선택",
                ["O", "X"],
                horizontal=True,
                key=f"answer_{i}",
                label_visibility="collapsed",
            )
            answers.append(choice)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("✅ 제출하고 채점", type="primary"):
            score = 0
            for i, (item, user_a) in enumerate(zip(st.session_state.questions, answers), start=1):
                if user_a == item["answer"]:
                    score += 1
            st.session_state.score = score
            st.session_state.submitted = True

    with colB:
        if st.button("🔄 다시 풀기"):
            st.session_state.submitted = False
            st.session_state.score = 0
            for i in range(1, len(st.session_state.questions) + 1):
                st.session_state[f"answer_{i}"] = "O"

    if st.session_state.submitted:
        st.success(f"점수: {st.session_state.score} / {len(st.session_state.questions)}")

        st.subheader("📌 해설 & 근거")
        for i, item in enumerate(st.session_state.questions, start=1):
            with st.expander(f"Q{i} 해설 보기"):
                st.markdown(f"- **정답:** {item['answer']}")
                if item.get("explain"):
                    st.markdown(f"- **해설:** {item['explain']}")
                if item.get("evidence"):
                    st.markdown(f"- **근거:** {item['evidence']}")
else:
    st.info("PDF를 업로드하고 인덱스를 만든 뒤 퀴즈를 생성해보세요.")
