# 파이썬 패키지 설치 명령어
# pip instal streamlit==1.52.2
# pip install fastapi==0.104.1
# pip install uvicorn==0.27.0.post1

# fastapi 웹서버 터미널 실행 명령어
# uvicorn src.server:app --reload

# server.py
from fastapi import FastAPI

app = FastAPI(title="Health-Eat API Server")

@app.get("/")
async def root():
    """
    루트 엔드포인트: Streamlit 프론트엔드에서 서버 연결 상태 확인 시 사용.
    """
    try:
        # 실제 환경에서는 DB 연결 확인이나 GPU 상태 등 체크 가능.
        return {
            "status": "success", 
            "message": "Health-Eat AI FastAPI 서버 정상 작동 중!",
            "version": "1.0"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# TODO: 필요 시 이 아래에 Rest API 추가 구현 예정.(2026.04.28 minjae)
# @app.post("/detect")
# async def detect_pill(file: UploadFile = File(...)):
#     ...