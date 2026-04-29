"""
streamlit 메인 웹페이지.

* streamlit 메인 웹페이지 -> 서브 웹페이지 이동
참고: https://leemcse.tistory.com/entry/%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99-main%EA%B3%BC-sub-%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99
"""

import streamlit as st
import requests
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSS_FILE_NAME = PROJECT_ROOT / "public" / "css" / "main.css"
FOOTER_FILE_NAME = PROJECT_ROOT / "public" / "images" / "footer_banner.png"

def load_css(file_name):
    """main.css 파일 로드 및 Streamlit 적용"""

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"[오류] main.css 파일 존재 안 함!: {file_name}")
    
def display_server_connection():
    """FastAPI 서버 연결 상태 확인 및 사이드바 표시."""
    st.sidebar.caption("서버 상태")
    
    try:
        # FastAPI 서버(기본 포트 8000) 루트 엔드포인트("/") Http GET 통신. (안정성을 위해 2초 타임아웃 설정)
        response = requests.get("http://127.0.0.1:8000/", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("🟢 서버 연결됨")
        else:
            st.sidebar.warning("🟡 서버 응답 오류")
            
    # except requests.exceptions.ConnectionError:
    except requests.exceptions.RequestException:
        st.sidebar.error("🔴 서버 꺼져있음")  # ConnectionError뿐만 아니라 Timeout 등 모든 요청 관련 에러 포괄하여 처리
    
def main_page():
    """Streamlit 메인 웹페이지"""
    try:
        # 아래 테스트 오류 코드 필요시 참고 (2025.10.30 minjae)
        # raise Exception('Streamlit 단위 기능 테스트')  # 예외 발생시킴
        print(f"PROJECT_ROOT: {str(PROJECT_ROOT)}")

        print("Streamlit 메인 웹페이지 함수 시작")

        # 1. 페이지 설정 (제목 및 아이콘)
        st.set_page_config(
            page_title="Health-Eat | 알약 탐지 서비스",
            page_icon="💊",
            layout="centered"
        )
        
        # ==========================================
        # 상태 관리 (Session State) 초기화
        # ==========================================
        # '닫기' 버튼을 눌렀을 때 파일 업로더를 강제로 비우기 위한 고유 키(Key)
        if 'uploader_key' not in st.session_state:
            st.session_state['uploader_key'] = 0
        
        # '탐지' 버튼을 눌렀을 때 메시지를 띄울지 여부를 결정하는 상태 값
        if 'show_detect_msg' not in st.session_state:
            st.session_state['show_detect_msg'] = False
        

        # 2. 커스텀 CSS (프로토타입 디자인 적용)
        load_css(str(CSS_FILE_NAME))

        # 3. 사이드바 (메뉴 및 서버 연결 상태 확인)
        st.sidebar.title("Menu")
        # menu = st.sidebar.radio("이동", ["🏠 홈", "📜 탐지 기록", "⚙️ 설정"])
        menu = st.sidebar.radio("이동", ["🏠 홈", "⚙️ 설정"])
        st.sidebar.markdown("---")

        display_server_connection()  # FastAPI 서버 통신 확인 로직 함수 호출

        # 4. 헤더 섹션
        # st.title("🍎 Health-Eat")
        st.title("🔍 PILL SIGHT 알약 탐지 프로그램")
        # st.subheader("🔍 PILL SIGHT 알약 탐지 프로그램")
        st.write("YOLO 기반 실시간 알약 탐지 및 분류 시스템 카메라 또는 이미지 입력으로 알약의 종류와 위치를 자동으로 감지하고, 실험별 성능 지표를 체계적으로 추적·비교합니다.")

        # 5. 검색창 섹션
        # st.write("") 
        # search_query = st.text_input("검색어 입력", placeholder="알약 이름을 입력하세요...", label_visibility="collapsed")
        # st.write("")

        # 6. 메인 기능 카드 (2열 레이아웃)
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("### 💊 알약 탐지")
                st.caption("AI Detection")
                st.write("사진 업로드 시 AI 모델이 알약을 탐지합니다.")
                
                # 버튼 클릭 시 액션 (나중에 FastAPI의 /detect API를 호출하게 됨)
                # if st.button("📷 사진 업로드"):
                #     st.info("알약 탐지 기능을 실행합니다... (API 연동 필요)")
                
                # 버튼 대신 파일 업로더 위젯 사용 (탐색기 열림, 파일 제한)
                uploaded_file = st.file_uploader(
                    "📷 사진 업로드", 
                    type=['jpg', 'jpeg', 'png', 'gif', 'webp'],
                    key=f"uploader_{st.session_state['uploader_key']}", # 키를 바꾸면 업로더가 초기화됨
                    label_visibility="collapsed" # 기본 라벨 글씨 숨김
                )

                if uploaded_file is None:
                    # 파일이 없을 때 기본 메시지
                    st.write("사진 업로드 시 AI 모델이 알약을 탐지합니다.")
                else:
                    # ==========================================
                    # 핵심 추가 코드: 이미지가 업로드되면 업로더 전체를 화면에서 숨깁니다!
                    # ==========================================
                    st.markdown("""
                        <style>
                        [data-testid="stFileUploader"] {
                            display: none !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    # ==========================================
                    
                    # 선택된 이미지 화면에 출력
                    st.image(uploaded_file, width="stretch")
                    
                    # 이미지가 있을 때만 "탐지", "닫기" 버튼 활성화
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        if st.button("탐지", width="stretch"):
                            st.session_state['show_detect_msg'] = True
                    
                    with btn_col2:
                        if st.button("닫기", width="stretch"):
                            # 닫기 누르면 이미지 초기화
                            # uploader_key 숫자를 1 올리면, Streamlit은 파일 업로더가 완전히 새로 생긴 줄 알고 안의 파일을 비워버립니다.
                            st.session_state['uploader_key'] += 1
                            st.session_state['show_detect_msg'] = False # 탐지 메시지도 함께 지움
                            st.rerun() # 즉시 화면 새로고침

                    # 탐지 버튼 클릭 시 메시지 출력
                    if st.session_state['show_detect_msg']:
                        st.info("[안내] 알약 탐지 기능 추후 구현 예정!")

        with col2:
            with st.container(border=True):
                st.markdown("### 📋 내 약 정보")
                st.caption("Medication Info")
                st.write("현재 복용 중인 약의 정보 및 스케줄 관리.")
                
                if st.button("🔍 약 정보 확인"):
                    st.info("내 약 정보를 불러옵니다... (API 연동 필요)")
                    
        # width="stretch" 옵션을 넣으면 상단 레이아웃 너비에 맞춰 100% 꽉 차게 확장됩니다.
        st.image(str(FOOTER_FILE_NAME), width="stretch")

        # 7. 간단한 푸터
        st.markdown("---")
        st.caption("© 2026 Health-Eat. All rights reserved.")
    except Exception as e:
        st.error(f"[오류] 기능 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main_page()