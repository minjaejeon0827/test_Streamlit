"""
FastAPI 서버와 Streamlit 앱 동시 실행.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "views"

# 전역 프로세스 리스트
processes = []

def signal_handler(signum, frame):
    """시그널 핸들러 - 프로세스 종료"""
    print("시그널 핸들러 호출")
    print("종료 신호 받음. 서버 종료...")
    
    for i, process in enumerate(processes):
        print(f"프로세스 {i+1} 종료 시작")
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"프로세스 {i+1} 정상 종료됨")
        except subprocess.TimeoutExpired:
            print(f"프로세스 {i+1} 강제 종료")
            process.kill()
        except Exception as e:
            print(f"프로세스 {i+1} 종료 중 오류: {e}")
    
    print("모든 서버 종료 완료.")
    sys.exit(0)

def main():
    print("Health-Eat 통합 서버 구동 시작")
    
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BACKEND_DIR: {BACKEND_DIR}")
    print(f"FRONTEND_DIR: {FRONTEND_DIR}")
    
    # 시그널 핸들러 등록
    print("시그널 핸들러 등록")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 백엔드 디렉토리에서 서버 실행
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    # Streamlit 이메일 묻는 화면을 강제로 무시하는 환경 변수 설정
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'

    # 1. FastAPI 백엔드 서버 백그라운드 실행
    # stdout=subprocess.PIPE 등을 쓰면 로그가 엉킬 수 있어 기본 출력으로 둡니다.
    process = subprocess.Popen(
        # [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=BACKEND_DIR,
        env=env,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    processes.append(process)
    print("FastAPI 프로세스 -> 프로세스 리스트 추가 완료")
    
    print("FastAPI 백엔드 서버 시작 요청 완료 (포트: 8000)")

    # 백엔드가 완전히 뜰 때까지 아주 잠깐 대기 (안정성을 위해)
    time.sleep(3)
    
    if process.poll() is None:
        print("FastAPI 서버 시작. (http://localhost:8000)")
        
    else:
        print("FastAPI 서버 시작 실패")

    # 2. Streamlit 프론트엔드 실행
    process = subprocess.Popen(
        # [sys.executable, "-m", "streamlit", "run", "home_page.py", 
        #  "--server.port", "8501",
        #  "--server.address", "0.0.0.0"],
        [sys.executable, "-m", "streamlit", "run", "home_page.py", 
         "--server.port", "8501",
         "--server.address", "127.0.0.1",
         "--server.headless", "true"], # 헤드리스 모드(이메일 생략) 추가
         cwd=FRONTEND_DIR,
         env=env,
         # stdout=subprocess.PIPE,
         # stderr=subprocess.STDOUT,
         universal_newlines=True
    )
        
    processes.append(process)
    print("Streamlit 프로세스 -> 프로세스 리스트 추가 완료")
    print("Streamlit 프론트엔드 시작 요청 완료")
    
    # Streamlit 프론트엔드 완전히 뜰 때까지 아주 잠깐 대기 (안정성을 위해)
    time.sleep(5)
    
    if process.poll() is None:
        print("Streamlit 앱 시작. (http://localhost:8501)")

    else:
        print("Streamlit 앱 시작 실패")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt 감지")
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()