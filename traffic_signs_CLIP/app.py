import os
import glob
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
import open_clip


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Traffic Sign CLIP Zero-shot Classifier",
    layout="wide",
)

# -----------------------------
# Utilities
# -----------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (M1/M2/M3/M4) 환경이면 아래가 True일 수 있음
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [N, D], b: [M, D] -> sim: [N, M]"""
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a @ b.T


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def load_image(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


# -----------------------------
# Default Prompt Set (Traffic Signs)
# 필요에 맞게 늘리면 됨
# -----------------------------
DEFAULT_CLASSES = [
    ("stop", [
        "a photo of a red octagonal stop sign",
        "a traffic sign that says STOP",
        "a stop sign on the road",
    ]),
    ("speed_limit", [
        "a photo of a speed limit sign",
        "a circular speed limit sign with a number",
        "a road sign indicating speed limit",
    ]),
    ("no_entry", [
        "a photo of a no entry sign",
        "a traffic sign with a red circle and a white horizontal bar",
        "a do not enter road sign",
    ]),
    ("yield", [
        "a photo of a yield sign",
        "a triangular yield sign",
        "a give way road sign",
    ]),
    ("pedestrian_crossing", [
        "a photo of a pedestrian crossing sign",
        "a road sign indicating pedestrian crossing",
        "a crosswalk warning sign",
    ]),
    ("traffic_light_ahead", [
        "a photo of a traffic light ahead warning sign",
        "a road sign indicating traffic signal ahead",
        "a sign warning of a traffic signal",
    ]),
    ("school_zone", [
        "a photo of a school zone sign",
        "a road sign indicating school zone",
        "a sign warning drivers to slow down near a school",
    ]),
    ("no_parking", [
        "a photo of a no parking sign",
        "a road sign indicating parking is not allowed",
        "a no parking traffic sign",
    ]),
    ("u_turn_prohibited", [
        "a photo of a no u-turn sign",
        "a road sign prohibiting u-turn",
        "a traffic sign with a u-turn arrow crossed out",
    ]),
    ("one_way", [
        "a photo of a one way sign",
        "a road sign indicating one-way traffic",
        "a one way arrow traffic sign",
    ]),
]

# 한국 도로표지에 맞게 프롬프트를 한국어로도 보강하고 싶으면 여기에 추가하세요.
KOREAN_HINTS = {
    "stop": ["정지 표지판 사진", "도로 정지 표지판"],
    "speed_limit": ["제한속도 표지판 사진", "속도 제한 표지"],
    "no_entry": ["진입금지 표지판 사진", "출입 금지 표지"],
    "yield": ["양보 표지판 사진", "서행 양보 표지"],
    "pedestrian_crossing": ["횡단보도 표지판 사진", "보행자 횡단 주의 표지"],
    "traffic_light_ahead": ["신호등 주의 표지판 사진", "전방 신호등 표지"],
    "school_zone": ["어린이 보호구역 표지판 사진", "스쿨존 표지"],
    "no_parking": ["주차금지 표지판 사진", "주정차 금지 표지"],
    "u_turn_prohibited": ["유턴금지 표지판 사진", "유턴 금지 표지"],
    "one_way": ["일방통행 표지판 사진", "일방통행 표지"],
}

# -----------------------------
# Model Loader (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_clip_model(model_name: str, pretrained: str, device_str: str):
    device = torch.device(device_str)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess, tokenizer


def build_text_features(
    model,
    tokenizer,
    class_prompts: Dict[str, List[str]],
    device: torch.device,
    normalize: bool = True,
) -> Tuple[List[str], torch.Tensor]:
    """
    class_prompts: {label: [prompt1, prompt2, ...]}
    return:
      labels: ["stop", "speed_limit", ...]
      text_feats: [num_labels, D]  (각 라벨의 여러 프롬프트 임베딩 평균)
    """
    labels = []
    feats = []
    with torch.no_grad():
        for label, prompts in class_prompts.items():
            tokens = tokenizer(prompts).to(device)
            tf = model.encode_text(tokens)  # [P, D]
            if normalize:
                tf = tf / (tf.norm(dim=-1, keepdim=True) + 1e-8)
            tf_mean = tf.mean(dim=0, keepdim=True)  # [1, D]
            if normalize:
                tf_mean = tf_mean / (tf_mean.norm(dim=-1, keepdim=True) + 1e-8)
            labels.append(label)
            feats.append(tf_mean)
    text_feats = torch.cat(feats, dim=0)  # [L, D]
    return labels, text_feats


def predict_image(
    model,
    preprocess,
    image: Image.Image,
    text_labels: List[str],
    text_feats: torch.Tensor,
    device: torch.device,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    img = load_image(image)
    img_t = preprocess(img).unsqueeze(0).to(device)  # [1, C, H, W]
    with torch.no_grad():
        img_feat = model.encode_image(img_t)  # [1, D]
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)

        sims = (img_feat @ text_feats.T).squeeze(0)  # [L]
        sims_np = sims.detach().float().cpu().numpy()
        probs = softmax(sims_np)  # pseudo-probabilities

    idx = np.argsort(-probs)[:topk]
    return [(text_labels[i], float(probs[i])) for i in idx]


# -----------------------------
# UI
# -----------------------------
st.title("🚦 Traffic Sign CLIP Zero-shot Classifier (Streamlit)")
st.caption("이미지 한 장 업로드 → CLIP으로 프롬프트 기반 분류 / 폴더 일괄 예측 지원")

with st.sidebar:
    st.subheader("⚙️ Model")
    # 가볍고 실용적인 조합 (CPU에서도 그나마 쓸만)
    model_name = st.selectbox(
        "CLIP 모델",
        ["ViT-B-32", "ViT-B-16", "ViT-L-14"],
        index=0,
    )
    pretrained = st.selectbox(
        "Pretrained",
        # open_clip에서 자주 쓰는 프리트레인
        ["openai", "laion2b_s34b_b79k", "laion2b_s32b_b82k"],
        index=0,
    )

    device = get_device()
    st.write(f"🖥️ Device: **{device.type}**")

    st.divider()
    st.subheader("🧠 Prompt / Classes")
    use_korean = st.checkbox("한국어 프롬프트도 함께 사용", value=True)

    # 기본 클래스 로드
    class_prompts = {}
    for label, prompts in DEFAULT_CLASSES:
        merged = list(prompts)
        if use_korean and label in KOREAN_HINTS:
            merged += KOREAN_HINTS[label]
        class_prompts[label] = merged

    # 사용자 커스텀 클래스 추가
    st.caption("원하면 클래스/프롬프트를 직접 추가할 수 있어요.")
    custom_block = st.text_area(
        "커스텀 클래스 (형식: label|prompt1;prompt2;prompt3)",
        value="",
        placeholder="예)\nparking|a photo of a parking sign; a road sign indicating parking\nwarning|a triangular warning road sign",
        height=120,
    )

    if custom_block.strip():
        for line in custom_block.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            label, prompts_str = line.split("|", 1)
            label = label.strip()
            prompts = [p.strip() for p in prompts_str.split(";") if p.strip()]
            if label and prompts:
                class_prompts[label] = prompts

    topk = st.slider("Top-K", 1, 10, 5)

# Load model
with st.spinner("CLIP 모델 로딩 중..."):
    model, preprocess, tokenizer = load_clip_model(model_name, pretrained, str(device))

# Build text features (cached by Streamlit session, but depends on prompts; keep it simple)
with st.spinner("텍스트 프롬프트 임베딩 생성 중..."):
    text_labels, text_feats = build_text_features(model, tokenizer, class_prompts, device)

tab1, tab2 = st.tabs(["🖼️ 단일 이미지", "📁 폴더 일괄 예측"])

with tab1:
    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        uploaded = st.file_uploader("교통표지판 이미지를 업로드하세요", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="입력 이미지", use_container_width=True)

    with colR:
        st.subheader("예측 결과")
        if uploaded:
            preds = predict_image(
                model=model,
                preprocess=preprocess,
                image=img,
                text_labels=text_labels,
                text_feats=text_feats,
                device=device,
                topk=topk,
            )
            df = pd.DataFrame(preds, columns=["label", "score"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            best_label, best_score = preds[0]
            st.success(f"✅ 예측: **{best_label}**  (score={best_score:.4f})")

            st.caption(
                "※ score는 softmax로 만든 '유사도 기반 확률처럼 보이는 값'입니다. "
                "정확한 확률이라기보다 클래스 간 상대 비교용으로 보세요."
            )
        else:
            st.info("왼쪽에서 이미지를 업로드하면 결과가 나옵니다.")

with tab2:
    st.subheader("📁 폴더 내 이미지 일괄 예측")
    st.caption("로컬 폴더 경로를 넣으면 JPG/PNG 등을 훑어서 Top-1 결과를 CSV로 내려받을 수 있어요.")

    folder = st.text_input("이미지 폴더 경로", value="", placeholder=r"C:\data\traffic_signs\images")
    exts = st.multiselect("확장자", ["jpg", "jpeg", "png", "webp"], default=["jpg", "jpeg", "png"])

    run = st.button("일괄 예측 실행", type="primary")

    if run:
        if not folder.strip():
            st.error("폴더 경로를 입력하세요.")
        elif not os.path.isdir(folder):
            st.error("폴더가 존재하지 않습니다.")
        else:
            patterns = []
            for e in exts:
                patterns.append(os.path.join(folder, f"**/*.{e}"))
                patterns.append(os.path.join(folder, f"**/*.{e.upper()}"))
            files = []
            for p in patterns:
                files.extend(glob.glob(p, recursive=True))
            files = sorted(list(set(files)))

            if not files:
                st.warning("해당 폴더에서 이미지 파일을 찾지 못했습니다.")
            else:
                st.write(f"총 파일 수: **{len(files)}**")
                rows = []
                prog = st.progress(0)
                t0 = time.time()

                for i, fp in enumerate(files, start=1):
                    try:
                        im = Image.open(fp)
                        pred1 = predict_image(
                            model=model,
                            preprocess=preprocess,
                            image=im,
                            text_labels=text_labels,
                            text_feats=text_feats,
                            device=device,
                            topk=1,
                        )[0]
                        rows.append({
                            "path": fp,
                            "pred_label": pred1[0],
                            "score": pred1[1],
                        })
                    except Exception as e:
                        rows.append({
                            "path": fp,
                            "pred_label": "ERROR",
                            "score": np.nan,
                            "error": str(e),
                        })

                    prog.progress(i / len(files))

                dt = time.time() - t0
                df = pd.DataFrame(rows)
                st.dataframe(df.head(50), use_container_width=True)

                st.success(f"완료 ✅ (처리시간: {dt:.2f}s, 평균 {dt/len(files):.4f}s/장)")

                csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "CSV 다운로드",
                    data=csv_bytes,
                    file_name="clip_batch_predictions.csv",
                    mime="text/csv",
                )

st.divider()
st.caption(
    "팁) 한국 교통표지판은 형태/색상 단서가 중요해서, "
    "프롬프트에 'red circle', 'blue circle', 'triangle', 'octagon', 'speed limit number' 같은 단서를 넣으면 성능이 좋아지는 경우가 많습니다."
)
