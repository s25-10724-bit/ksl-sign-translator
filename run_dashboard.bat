@echo off
echo ================================
echo KSL Sign Translator Dashboard
echo ================================
echo.

python --version
if errorlevel 1 (
    echo Python이 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo https://www.python.org 에서 Python을 설치하세요.
    pause
    exit /b
)

echo.
echo 필요한 라이브러리를 설치합니다.
pip install -r requirements.txt

echo.
echo 대시보드를 실행합니다.
echo 브라우저가 열리면 실시간 인식 시작 버튼을 누르세요.
echo 종료하려면 이 창에서 Ctrl + C를 누르세요.
echo.

streamlit run dashboard.py

pause
