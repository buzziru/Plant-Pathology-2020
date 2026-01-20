import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import tempfile
import os
import pandas as pd


# 페이지 및 스타일 설정
st.set_page_config(
    page_title="Plant Pathology 2020 AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# HF Space API 주소
HF_API_URL = "https://ingyoun-plant-pathology-api.hf.space"

# 샘플 이미지 경로 리스트
SAMPLE_IMAGES = {
    "Sample 1": "./samples/Test_0.jpg",
    "Sample 2": "./samples/Test_1.jpg",
    "Sample 3": "./samples/Test_2.jpg",
    "Sample 4": "./samples/Test_3.jpg", 
    "Sample 5": "./samples/Test_4.jpg", 
}


@st.cache_data(show_spinner=False)
def call_api(file_path):
    """
    Hugging Face API를 호출하고 결과를 캐싱합니다.
    동일한 이미지에 대해 중복 호출을 방지합니다.
    """
    try:
        client = Client(HF_API_URL)
        result = client.predict(
            image=handle_file(file_path),
            api_name="/predict"
        )
        return result
    except Exception as e:
        return {"error": str(e)}

def save_uploaded_file(uploaded_file):
    """
    업로드 객체를 임시 파일로 저장하고 경로를 반환합니다.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name
    except Exception as e:
        st.error(f"파일 저장 중 오류 발생: {e}")
        return None
    
def reset_state():
    """상태 초기화 (처음으로 돌아가기)"""
    st.session_state['current_image_path'] = None
    st.session_state['current_image_obj'] = None
    st.rerun()
    

# 세션 상태 초기화
if 'current_image_path' not in st.session_state:
    st.session_state['current_image_path'] = None
if 'current_image_obj' not in st.session_state:
    st.session_state['current_image_obj'] = None


# 메인 레이아웃
st.title("🌿 Plant Pathology 2020 식물 병해 진단")
st.markdown("식물 잎 사진을 업로드하거나 샘플을 선택하여 병해를 진단하세요.")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

# 이미지 입력 및 표시
with col_left:
    st.subheader("1. 이미지 입력")
    uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        temp_path = save_uploaded_file(uploaded_file)
        st.session_state['current_image_path'] = temp_path
        st.session_state['current_image_obj'] = Image.open(uploaded_file)

    # 현재 선택된 이미지 표시
    if st.session_state['current_image_obj']:
        st.image(
            st.session_state['current_image_obj'], 
            caption="분석 대상 이미지", 
            use_container_width=True
        )
    else:
        st.info("이미지를 업로드하거나 아래 샘플을 선택해주세요.")
        st.empty()
        
    if st.button("🔄 다른 이미지 분석하기"):
        reset_state()

# 추론 결과 표시
with col_right:
    st.subheader("2. AI 진단 결과")

    if st.session_state['current_image_path']:
        with st.spinner("AI가 잎의 상태를 정밀 분석 중입니다..."):
            api_result = call_api(st.session_state['current_image_path'])

        if "error" in api_result:
            st.error(f"API 호출 실패: {api_result['error']}")
        else:
            # 포맷: {'label': 'Rust', 'confidences': [{'label': 'Rust', 'confidence': 0.98}, ...]}
            top_label = api_result.get('label', 'Unknown')
            confidences = api_result.get('confidences', [])

            # 데이터프레임 변환 및 정렬 (확률 내림차순)
            df_res = pd.DataFrame(confidences)
            if not df_res.empty:
                df_res = df_res.sort_values(by='confidence', ascending=False)
 
                # 가장 높은 확률 강조 (Metric)
                top_conf = df_res.iloc[0]['confidence']
                st.metric(
                    label="가장 유력한 진단명", 
                    value=top_label, 
                    delta=f"{top_conf:.1%}"
                )

                # 전체 확률 차트
                st.markdown("### 상세 확률 분포")
                for _, row in df_res.iterrows():
                    label_name = row['label']
                    score = row['confidence']
                    st.write(f"**{label_name}** ({score:.1%})")
                    st.progress(score)
            else:
                st.warning("결과 데이터 형식이 예상과 다릅니다.")

    else:
        st.write("이미지가 준비되면 이곳에 결과가 표시됩니다.")

st.divider()
st.subheader("💡 샘플 이미지로 테스트하기")

valid_samples = {name: path for name, path in SAMPLE_IMAGES.items() if os.path.exists(path)}

if valid_samples:
    cols = st.columns(len(valid_samples))
    for idx, (name, path) in enumerate(valid_samples.items()):
        with cols[idx]:
            img = Image.open(path)
            st.image(img, use_container_width=True)

            if st.button(f"{name} 선택", key=f"btn_{idx}"):
                st.session_state['current_image_path'] = path
                st.session_state['current_image_obj'] = img
else:
    st.caption("samples 폴더에 이미지를 넣어두면 여기에 표시됩니다.")
