@echo off
setlocal

cd /d "%~dp0"

set "APP_URL=http://localhost:8501"
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "DRY_RUN="
set "RUN_CMD="

if /i "%~1"=="--dry-run" set "DRY_RUN=1"

call :resolve_bootstrap_python
if errorlevel 1 goto :error

if not exist "%VENV_PY%" (
  echo [1/5] Creating virtual environment...
  if defined DRY_RUN (
    echo [DRY RUN] %BOOTSTRAP_CMD% -m venv "%VENV_DIR%"
  ) else (
    call %BOOTSTRAP_CMD% -m venv "%VENV_DIR%" || goto :error
  )
)

if exist "%VENV_PY%" (
  set "RUN_CMD=\"%VENV_PY%\""
) else (
  set "RUN_CMD=%BOOTSTRAP_CMD%"
)

echo [2/5] Checking required packages...
if defined DRY_RUN (
  echo [DRY RUN] %RUN_CMD% -c "import streamlit, easyocr, cv2, yt_dlp, numpy"
) else (
  call %RUN_CMD% -c "import streamlit, easyocr, cv2, yt_dlp, numpy" >nul 2>nul
  if errorlevel 1 (
    echo [3/5] Installing required packages...
    call %RUN_CMD% -m pip install -r requirements.txt || goto :error
  ) else (
    echo Dependencies are already installed.
  )
)

echo [4/5] Opening browser...
if defined DRY_RUN (
  echo [DRY RUN] start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process '%APP_URL%'"
  echo [DRY RUN] %RUN_CMD% -m streamlit run web_ui.py
  goto :success
)

start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process '%APP_URL%'"

echo [5/5] Starting Web UI...
echo If the browser does not open automatically, open %APP_URL%.
echo Press Ctrl+C in this window to stop the app.
call %RUN_CMD% -m streamlit run web_ui.py
goto :success

:resolve_bootstrap_python
where py >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP_CMD=py -3"
  exit /b 0
)

where python >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP_CMD=python"
  exit /b 0
)

echo Python 3.11 or later was not found.
exit /b 1

:error
echo.
echo Startup failed.
echo Run "pip install -r requirements.txt" in PowerShell if needed.
pause
exit /b 1

:success
endlocal
exit /b 0
