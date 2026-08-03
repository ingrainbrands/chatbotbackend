@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo Starting FastAPI Backend Server on http://localhost:8000...
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

