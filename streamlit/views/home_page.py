# streamlit 메인 웹페이지 -> 서브 웹페이지 이동
# 참고 URL - https://leemcse.tistory.com/entry/%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99-main%EA%B3%BC-sub-%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99

# home_page.py
import streamlit as st
import requests

# 1. 페이지 설정 (제목 및 아이콘)
st.set_page_config(
    page_title="Health-Eat | 알약 인식 서비스",
    page_icon="💊",
    layout="centered"
)

# 2. 커스텀 CSS (프로토타입 디자인 적용)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ddd;
        font-weight: bold;
    }
    .stButton>button:hover {
        border-color: #007bff;
        color: #007bff;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stHeader) {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 사이드바 (메뉴 및 서버 연결 상태 확인)
st.sidebar.title("Menu")
menu = st.sidebar.radio("이동", ["🏠 홈", "📜 인식 기록", "⚙️ 설정"])
st.sidebar.markdown("---")

# 🌟 백엔드(FastAPI) 통신 확인 로직
st.sidebar.caption("서버 상태")
try:
    # FastAPI 서버(기본 포트 8000)의 루트 엔드포인트를 찔러봅니다.
    response = requests.get("http://127.0.0.1:8000/")
    if response.status_code == 200:
        st.sidebar.success("🟢 백엔드 연결됨")
    else:
        st.sidebar.warning("🟡 백엔드 응답 오류")
except requests.exceptions.ConnectionError:
    st.sidebar.error("🔴 백엔드 꺼져있음")


# 4. 헤더 섹션
st.title("🍎 Health-Eat")
st.subheader("건강하고 안전한 약 복용을 위한 알약 인식 서비스")

# 5. 검색창 섹션
st.write("") 
search_query = st.text_input("검색어 입력", placeholder="알약 이름을 입력하세요...", label_visibility="collapsed")
st.write("")

# 6. 메인 기능 카드 (2열 레이아웃)
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 💊 알약 인식")
        st.caption("AI Detection")
        st.write("사진을 업로드하거나 촬영하면 AI가 알약을 분석합니다.")
        
        # 버튼 클릭 시 액션 (나중에 FastAPI의 /detect API를 호출하게 됨)
        if st.button("📷 사진 업로드 / 촬영"):
            st.info("알약 인식 기능을 실행합니다... (API 연동 필요)")

with col2:
    with st.container(border=True):
        st.markdown("### 📋 내 약 정보")
        st.caption("Medication Info")
        st.write("현재 복용 중인 약의 정보와 스케줄을 관리합니다.")
        
        if st.button("🔍 약 정보 확인"):
            st.info("내 약 정보를 불러옵니다... (API 연동 필요)")

# 7. 간단한 푸터
st.markdown("---")
st.caption("© 2026 Health-Eat. All rights reserved.")