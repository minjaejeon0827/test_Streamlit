# 파이썬 패키지 설치 명령어
# pip instal streamlit==1.52.2
# pip install fastapi==0.104.1
# pip install uvicorn==0.27.0.post1

# fastapi 웹서버 터미널 실행 명령어
# uvicorn src.server:app --reload

# server.py
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="Health-Eat API Server")

# 허용할 안전한 이미지 포맷 (Pillow가 인식하는 포맷 이름)
ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG", "WEBP"]

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

@app.post("/detect")
async def detect_pill(file: UploadFile = File(...)):
    """
    강력한 시큐어 코딩(이미지 재가공)이 적용된 탐지 엔드포인트
    """
    try:
        # 1. 클라이언트가 보낸 파일을 메모리에서 읽어옵니다. (아직 서버 하드디스크에 저장 안 함!)
        contents = await file.read()
        
        # ==========================================
        # 🛡️ 시큐어 코딩 2단계: 이미지 재가공 (Re-encoding) 및 스텔스 파일 정화
        # ==========================================
        try:
            # 해커가 보낸 파일(곰 인형)을 Pillow 공장에 넣어서 엽니다.
            # (만약 여기서 그림 파일이 아니라 그냥 악성 스크립트면 바로 에러가 나면서 튕겨 나갑니다)
            img = Image.open(io.BytesIO(contents))
            
            # 파일의 진짜 포맷이 우리가 허락한 것(JPEG, PNG 등)인지 확인합니다.
            if img.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=400, detail=f"위험 감지! 허용되지 않은 포맷({img.format})입니다.")

            # 🌟 핵심: 깨끗한 새 도화지(메모리 공간)를 준비합니다.
            secure_image_io = io.BytesIO()
            
            # 원래 이미지의 생김새만 그대로 새 도화지에 다시 그려서(저장해서) 덮어씌웁니다!
            # 이때 끝에 몰래 붙어있던 악성 스크립트(PHP 등)는 그림 데이터가 아니기 때문에 전부 버려집니다.
            img.save(secure_image_io, format=img.format)
            
            # 이제 불순물이 완벽히 제거된 100% 순수 이미지 데이터만 남았습니다.
            clean_image_bytes = secure_image_io.getvalue()
            
        except UnidentifiedImageError:
            # 아예 열리지도 않는 가짜 이미지(실행 파일 위장 등)를 잡아냅니다.
            raise HTTPException(status_code=400, detail="위험 감지! 손상되었거나 이미지가 아닌 파일입니다.")
        # ==========================================

        # 철통 방어 검증 통과! 
        # 이제 이 깨끗한 clean_image_bytes를 AI(YOLO 모델)에게 넘겨서 탐지하면 됩니다.
        
        file_name = file.filename
        
        return {
            "status": "success",
            "message": f"'{file_name}' 시큐어 검증 및 정화 완료! 안전하게 AI 탐지를 시작합니다.",
            "detected_pills": ["타이레놀(예시)"] 
        }
        
    except HTTPException as http_exc:
        # 위에서 발생시킨 400 에러를 프론트엔드로 전달
        raise http_exc
    except Exception as e:
        return {"status": "error", "message": f"서버 내부 오류: {str(e)}"}