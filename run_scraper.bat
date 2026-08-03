@echo off
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting Iryax Scraper silently in background... Logging to scraper.log
start "" /B venv\Scripts\pythonw.exe scraper.py





