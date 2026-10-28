@echo off
rem Double-click to build a standalone Terrasort.exe (needs Python + internet, takes a few minutes).
cd /d "%~dp0"
title Building Terrasort.exe
where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
%PY% -m pip install -r requirements.txt pyinstaller
if not %errorlevel%==0 goto fail
%PY% -m PyInstaller --noconfirm --clean --onedir --name Terrasort ^
  --add-data "terrasort\static;terrasort\static" ^
  --collect-submodules uvicorn --collect-submodules anyio --collect-submodules fastapi --collect-submodules starlette ^
  --collect-all laspy --collect-all lazrs --collect-all ezdxf ^
  terrasort_app.py
if not %errorlevel%==0 goto fail
echo.
echo Done. Your program is: %~dp0dist\Terrasort\Terrasort.exe
echo Copy the whole "dist\Terrasort" folder wherever you like; double-click Terrasort.exe to start.
pause
exit /b 0
:fail
echo.
echo Build failed - copy the messages above and send them to Claude.
pause
exit /b 1
