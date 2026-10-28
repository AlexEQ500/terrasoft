@echo off
rem Double-click to start Terrasort. Keep this window open while you use it.
cd /d "%~dp0"
title Terrasort
where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
%PY% -c "import numpy, scipy, laspy, fastapi, uvicorn, ezdxf" >nul 2>nul
if not %errorlevel%==0 (
  echo Installing required libraries, one-off, needs internet...
  %PY% -m pip install -r requirements.txt
)
rem Stop any older Terrasort still holding the port, so the newest version always runs
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>nul
echo Starting Terrasort - your browser will open at the newest version automatically.
echo Close this window to stop it.
%PY% terrasort_app.py %*
echo.
echo Terrasort stopped.
pause
