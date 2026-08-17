@echo off
title DocIntel Platform Launcher
cd /d "%~dp0"
echo =========================================================
echo Launching DocIntel Platform...
echo =========================================================
echo.
echo Starting Streamlit Web Dashboard...
echo Access the app at: http://localhost:8501
echo.
".venv\Scripts\python.exe" -m streamlit run app/ui/main.py
pause
