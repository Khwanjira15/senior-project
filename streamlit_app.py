"""Streamlit interface for the chest X-ray model demo."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from inference_service import (
    ATTENTION_TYPES,
    MODEL_OPTIONS,
    get_available_model_configs,
    get_model_label,
    get_recommended_model_config,
    predict_image,
    save_named_bytes_to_temp,
)


st.set_page_config(
    page_title="Clinical AI Interface",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(31, 122, 99, 0.14), transparent 28%),
              radial-gradient(circle at right top, rgba(113, 164, 145, 0.18), transparent 24%),
              linear-gradient(180deg, #eef5f1 0%, #dce9e2 100%);
          }
          .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1260px;
          }
          .hero-card, .panel-card, .metric-card, .result-card, .cam-card {
            background: rgba(250, 254, 251, 0.88);
            border: 1px solid rgba(23, 51, 43, 0.10);
            border-radius: 28px;
            box-shadow: 0 28px 70px rgba(18, 57, 46, 0.10);
          }
          .hero-card {
            padding: 2rem 2rem 1.7rem 2rem;
            margin-bottom: 1.2rem;
          }
          .eyebrow {
            margin: 0;
            color: #145845;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.76rem;
            font-weight: 700;
          }
          .hero-title {
            margin: 0.45rem 0 0 0;
            font-size: 3rem;
            line-height: 1.03;
            color: #17332b;
            font-weight: 700;
          }
          .hero-copy {
            margin-top: 1rem;
            max-width: 52rem;
            color: #55716a;
            font-size: 1.03rem;
            line-height: 1.75;
          }
          .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.2rem;
          }
          .metric-card {
            padding: 1.1rem 1.2rem;
          }
          .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #145845;
            line-height: 1;
          }
          .metric-label {
            display: block;
            margin-top: 0.35rem;
            color: #55716a;
            font-size: 0.92rem;
          }
          .section-title {
            margin: 0 0 0.75rem 0;
            color: #17332b;
            font-size: 1.3rem;
            font-weight: 700;
          }
          .muted {
            color: #55716a;
            line-height: 1.6;
          }
          .sidebar-note {
            background: linear-gradient(180deg, rgba(217, 239, 230, 0.85), rgba(245, 251, 248, 0.92));
            border: 1px solid rgba(31, 122, 99, 0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-top: 1rem;
            color: #55716a;
            font-size: 0.92rem;
            line-height: 1.55;
          }
          .result-card {
            padding: 1.4rem;
            margin-bottom: 1rem;
          }
          .result-title {
            margin: 0;
            color: #17332b;
            font-size: 2rem;
            line-height: 1.02;
            font-weight: 800;
          }
          .result-meta {
            margin-top: 0.5rem;
            color: #55716a;
            font-size: 1rem;
          }
          .cam-card {
            padding: 1rem;
            margin-top: 0.5rem;
          }
          .cam-label {
            color: #145845;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.65rem;
          }
          div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px dashed rgba(31, 122, 99, 0.32);
            border-radius: 18px;
          }
          div[data-testid="stSelectbox"] > div {
            border-radius: 14px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def data_url_to_bytes(data_url: str) -> bytes:
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
          <p class="eyebrow">Clinical AI Interface</p>
          <h1 class="hero-title">ระบบวิเคราะห์ภาพทรวงอกด้วย AI พร้อมผลอธิบายการตัดสินใจของโมเดล</h1>
          <p class="hero-copy">
            ออกแบบสำหรับการสาธิตงานวิจัยทางการแพทย์ โดยให้ผู้ใช้เลือก backbone model,
            เปรียบเทียบ baseline กับ attention mechanism และดูผลอธิบายเชิงภาพจาก CAM ในหน้าเดียวอย่างเป็นระเบียบ
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-value">6</div><span class="metric-label">Backbone Models</span></div>
          <div class="metric-card"><div class="metric-value">6</div><span class="metric-label">Attention Modes</span></div>
          <div class="metric-card"><div class="metric-value">3</div><span class="metric-label">Explainability Maps</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_deployment_help():
    recommended = get_recommended_model_config()
    st.error("แอปออนไลน์ยังไม่มีไฟล์น้ำหนักโมเดล จึงยังไม่สามารถรันผลทำนายได้ค่ะ")

    if recommended:
        st.markdown(
            f"""
            <div class="sidebar-note">
              <strong>ไฟล์ที่แนะนำให้อัปขึ้น GitHub ก่อน</strong><br>
              {recommended["checkpoint_path"].as_posix()}<br><br>
              เมื่อ push ไฟล์นี้ขึ้น repository แล้ว Streamlit Cloud จะสามารถรันชุด
              <strong>{recommended["model_label"]} + {recommended["attention_type"]}</strong> ได้ทันที
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-note">
              <strong>ยังไม่พบ checkpoint ในโปรเจคนี้</strong><br>
              ต้องเพิ่มไฟล์น้ำหนักโมเดลอย่างน้อย 1 ชุด เช่น
              <code>Results_PyTorch_Attention4/MobileNet_ECA/MobileNet_final.pth</code>
              หรือ <code>Results_PyTorch_Baseline4/MobileNet_Baseline/MobileNet_final.pth</code>
              ก่อนแอปออนไลน์จะรัน inference ได้
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_batch_prediction(uploaded_files, model_name: str, attention_type: str):
    results = []
    for uploaded_file in uploaded_files:
        image_path = save_named_bytes_to_temp(uploaded_file.name, uploaded_file.getvalue())
        result = predict_image(image_path, model_name, attention_type)
        result.filename = uploaded_file.name
        results.append(result)
    return results


def render_probability_table(probabilities):
    rows = []
    for label, value in probabilities.items():
        rows.append({"Class": label, "Probability": f"{value * 100:.2f}%"})
    st.dataframe(rows, width="stretch", hide_index=True)


def render_result(result):
    st.markdown(
        f"""
        <div class="result-card">
          <div class="eyebrow">Prediction Summary</div>
          <h2 class="result-title">{result.predicted_class}</h2>
          <div class="result-meta">{result.filename} • Confidence {result.confidence * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Class Probabilities</p>', unsafe_allow_html=True)
    render_probability_table(result.probabilities)

    st.markdown('<p class="section-title">Explainability Maps</p>', unsafe_allow_html=True)
    st.caption("ภาพ CAM ถูกแสดงในขนาดจำกัดเพื่อรักษาความคมชัดและลดการซูมเกินความละเอียดจริง")
    cam_columns = st.columns(2, gap="medium")
    with cam_columns[0]:
        st.markdown('<div class="cam-card"><div class="cam-label">Input Preview</div>', unsafe_allow_html=True)
        st.image(data_url_to_bytes(result.preview_image), width=420)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="cam-card"><div class="cam-label">GradCAM++</div>', unsafe_allow_html=True)
        st.image(data_url_to_bytes(result.gradcampp), width=420)
        st.markdown("</div>", unsafe_allow_html=True)

    with cam_columns[1]:
        st.markdown('<div class="cam-card"><div class="cam-label">GradCAM</div>', unsafe_allow_html=True)
        st.image(data_url_to_bytes(result.gradcam), width=420)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="cam-card"><div class="cam-label">ScoreCAM</div>', unsafe_allow_html=True)
        st.image(data_url_to_bytes(result.scorecam), width=420)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    inject_styles()
    render_hero()
    available_configs = get_available_model_configs()
    available_models = []
    for config in available_configs:
        if config["model_name"] not in available_models:
            available_models.append(config["model_name"])

    with st.sidebar:
        st.markdown("## Configuration")
        st.markdown(
            '<p class="muted">เลือกไฟล์ภาพหลายรูปพร้อมกัน เลือกโมเดลและ attention จากนั้นกดรันเพื่อดูผลแบบเปรียบเทียบได้ทันที</p>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "อัปโหลดภาพหลายไฟล์",
            type=[
                "png",
                "jpg",
                "jpeg",
                "jpe",
                "jfif",
                "bmp",
                "dib",
                "tif",
                "tiff",
                "webp",
                "gif",
                "heic",
                "heif",
                "avif",
                "pdf",
                "dcm",
                "dicom",
                "ima",
            ],
            accept_multiple_files=True,
        )

        if available_models:
            recommended = get_recommended_model_config()
            default_model_index = 0
            if recommended and recommended["model_name"] in available_models:
                default_model_index = available_models.index(recommended["model_name"])

            model_name = st.selectbox(
                "เลือกโมเดล",
                options=available_models,
                index=default_model_index,
                format_func=get_model_label,
            )
            available_attentions = [
                attention
                for attention in ATTENTION_TYPES
                if any(
                    config["model_name"] == model_name and config["attention_type"] == attention
                    for config in available_configs
                )
            ]
            default_attention_index = 0
            if recommended and recommended["model_name"] == model_name:
                if recommended["attention_type"] in available_attentions:
                    default_attention_index = available_attentions.index(recommended["attention_type"])

            attention_type = st.selectbox(
                "เลือก Attention",
                options=available_attentions,
                index=default_attention_index,
            )
            st.caption(f"พร้อมใช้งานบนเครื่อง/เซิร์ฟเวอร์นี้ {len(available_configs)} ชุดโมเดล")
        else:
            model_name = None
            attention_type = None
            render_deployment_help()

        st.markdown(
            '<div class="sidebar-note"><strong>Supported Inputs</strong><br>PNG, JPG, TIFF, BMP, WEBP, HEIC, PDF และ DICOM (.dcm, .dicom, .ima)</div>',
            unsafe_allow_html=True,
        )

        run_clicked = st.button("Run Inference", width="stretch", type="primary", disabled=not available_configs)

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []

    if run_clicked:
        if not uploaded_files:
            st.warning("กรุณาอัปโหลดอย่างน้อย 1 ไฟล์ก่อนค่ะ")
        else:
            try:
                with st.spinner("กำลังรันโมเดลและสร้าง CAM สำหรับหลายไฟล์อยู่ค่ะ..."):
                    st.session_state.batch_results = run_batch_prediction(uploaded_files, model_name, attention_type)
                st.success(f"รันเสร็จแล้วทั้งหมด {len(st.session_state.batch_results)} ไฟล์")
            except FileNotFoundError as exc:
                st.session_state.batch_results = []
                st.error(str(exc))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.session_state.batch_results = []
                st.error(f"เกิดข้อผิดพลาดระหว่างรันโมเดล: {exc}")

    if not st.session_state.batch_results:
        st.info("อัปโหลดภาพแล้วกด Run Inference เพื่อเริ่มดูผลลัพธ์ค่ะ")
        return

    st.markdown('<p class="section-title">Batch Results</p>', unsafe_allow_html=True)
    file_labels = [
        f"{result.filename} • {result.predicted_class} • {result.confidence * 100:.2f}%"
        for result in st.session_state.batch_results
    ]
    selected_label = st.radio(
        "เลือกรูปที่ต้องการดูรายละเอียด",
        options=file_labels,
        horizontal=False,
        label_visibility="collapsed",
    )
    selected_index = file_labels.index(selected_label)
    render_result(st.session_state.batch_results[selected_index])


if __name__ == "__main__":
    main()
